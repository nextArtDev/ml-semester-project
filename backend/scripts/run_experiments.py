"""Batch experiment runner.

Runs the full experimental matrix described in the project proposal:

* an **exact baseline** for every training-set size, and
* an **HNSW sweep** over a list of ``ef_search`` values for every size.

Every run is executed through the same ``run_experiment`` service the API
uses, so the numbers produced here are identical to the numbers the web
interface reports.

Guarantees enforced by this script:

* The exact and HNSW arms of a size share one training pool, one query set,
  one fitted preprocessing state, one metric and one ``k``.
* The training pool is subsampled **stratified by label**, with a seeded
  ``setseed`` so the same ``--seed`` reproduces the same pool.
* Repetitions are real: the whole timed query set is replayed
  ``--repetitions`` times and every sample is stored.
* A HNSW row whose execution plan did **not** use the index is printed as a
  warning, because a silent sequential-scan fallback would otherwise look
  like a very slow HNSW result.

Usage::

    python -m scripts.run_experiments --dataset-id 1 \
        --sizes 10000,50000 --ef-search 10,50,100 --repetitions 3
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.database import execute, fetch_one  # noqa: E402
from app.services import vector_store as vs  # noqa: E402
from app.services.classification import repetition_summary  # noqa: E402
from app.services.experiments import (  # noqa: E402
    ExperimentConfig,
    get_experiment,
    get_repetitions,
    run_experiment,
)
from app.services.metrics import DistanceMetric  # noqa: E402

FIELDS = [
    "dataset_id",
    "train_size",
    "test_size",
    "unique_queries",
    "total_test_pool",
    "retrieval_method",
    "k",
    "distance_metric",
    "hnsw_ef_search",
    "hnsw_m",
    "hnsw_ef_construction",
    "repetitions",
    "measured_queries",
    "preprocessing_mode",
    "recall_at_k",
    "tie_aware_recall_at_k",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "macro_f1",
    "accuracy_std",
    "macro_f1_std",
    "recall_at_k_std",
    "tie_aware_recall_at_k_std",
    "latency_avg_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "latency_min_ms",
    "latency_max_ms",
    "index_build_ms",
    "index_created_ms",
    "index_reused",
    "index_reused_build_ms",
    "index_size_bytes",
    "tied_queries",
    "max_tied_candidates",
    "dataset_fingerprint",
    "seed",
    "speedup_p50",
    "speedup_avg",
    "used_index_scan",
    "used_sequential_scan",
    "applied_ef_search",
    "api_overhead_s",
    "total_runtime_s",
    "env_python",
    "env_os",
    "env_machine",
    "env_cpu_cores",
    "env_postgresql",
    "env_pgvector",
]

REPETITION_FIELDS = [
    "experiment_id",
    "dataset_id",
    "train_size",
    "retrieval_method",
    "ef_search",
    "repetition",
    "unique_queries",
    "samples",
    "latency_mean_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "recall_at_k",
    "tie_aware_recall_at_k",
    "accuracy",
    "precision_macro",
    "recall_macro",
    "macro_f1",
]

SUMMARY_FIELDS = [
    "experiment_id",
    "dataset_id",
    "train_size",
    "retrieval_method",
    "ef_search",
    "repetitions",
    "metric",
    "mean",
    "median",
    "sd",
    "min",
    "max",
    "ci95_low",
    "ci95_high",
]

ENV_KEYS = {
    "env_python": "python",
    "env_os": "os",
    "env_machine": "machine",
    "env_cpu_cores": "cpu_cores",
    "env_postgresql": "postgresql",
    "env_pgvector": "pgvector",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the exact / HNSW experiment matrix for one dataset."
    )
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument(
        "--sizes",
        default="",
        help="Comma-separated training-set sizes to evaluate. "
        "Empty means use the dataset as-is.",
    )
    parser.add_argument(
        "--ef-search",
        default="10,25,50,100,200,400",
        help="Comma-separated hnsw.ef_search values for the sweep.",
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", choices=["l2", "cosine", "l1"], default="l2")
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument(
        "--repetitions",
        type=int,
        default=3,
        help="How many times to replay the timed query set (default 3).",
    )
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the stratified training-pool subsampling.",
    )
    parser.add_argument("--output", default="", help="CSV output path")
    parser.add_argument(
        "--json", dest="json_output", default="", help="JSON output path"
    )
    parser.add_argument(
        "--repetition-details",
        default="",
        help="Per-repetition CSV output path (one row per experiment and repetition).",
    )
    parser.add_argument(
        "--repetition-summary",
        default="",
        help="Cross-repetition aggregate CSV output path (median/dispersion per metric).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training-pool management
# ---------------------------------------------------------------------------
def snapshot_pool(dataset_id: int) -> list[tuple]:
    with vs.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                vs.sql.SQL(
                    "SELECT id, row_index, label, embedding::text FROM {t} ORDER BY id"
                ).format(t=vs.sql.Identifier(vs.train_table(dataset_id)))
            )
            return list(cur.fetchall())


def restore_pool(dataset_id: int, rows: list[tuple]) -> None:
    """Restore the full training pool from a snapshot and drop the index."""
    table = vs.train_table(dataset_id)
    with vs.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                vs.sql.SQL("TRUNCATE TABLE {t}").format(t=vs.sql.Identifier(table))
            )
            if rows:
                with cur.copy(
                    vs.sql.SQL(
                        "COPY {t} (id, row_index, label, embedding) FROM STDIN"
                    ).format(t=vs.sql.Identifier(table))
                ) as copy:
                    for row in rows:
                        copy.write_row(
                            (
                                int(row["id"]),
                                int(row["row_index"]),
                                int(row["label"]),
                                row["embedding"],
                            )
                        )
    _drop_index(dataset_id)


def subset_dataset(dataset_id: int, max_train_rows: int | None, seed: int) -> int | None:
    """Reduce the training pool to ``max_train_rows`` rows, stratified by label.

    Rows are removed from the database (not merely ignored) so that the
    measured latency reflects the real table size. Sampling is proportional
    within each class, with a session-level ``setseed`` for reproducibility.

    Returns the resulting row count, or ``None`` when no reduction happened.
    """
    if max_train_rows is None:
        return None

    table = vs.train_table(dataset_id)
    with vs.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                vs.sql.SQL("SELECT count(*) AS c FROM {t}").format(
                    t=vs.sql.Identifier(table)
                )
            )
            total = int(cur.fetchone()["c"])
            if total <= max_train_rows:
                return total

            fraction = max_train_rows / total
            # setseed must run in the same session as the random() call.
            cur.execute("SELECT setseed(%s)", (seed / 10000.0,))
            cur.execute(
                vs.sql.SQL(
                    "WITH ranked AS ("
                    "  SELECT id,"
                    "         row_number() OVER ("
                    "           PARTITION BY label ORDER BY random()"
                    "         ) AS rn,"
                    "         count(*) OVER (PARTITION BY label) AS cnt"
                    "  FROM {t}"
                    "), keep AS ("
                    "  SELECT id FROM ranked"
                    "  WHERE rn <= GREATEST(1, floor(cnt * %s))"
                    ")"
                    "DELETE FROM {t} WHERE id NOT IN (SELECT id FROM keep)"
                ).format(t=vs.sql.Identifier(table)),
                (fraction,),
            )
            cur.execute(
                vs.sql.SQL("SELECT count(*) AS c FROM {t}").format(
                    t=vs.sql.Identifier(table)
                )
            )
            kept = int(cur.fetchone()["c"])

    _drop_index(dataset_id)
    print(f"  stratified subsample: {total} -> {kept} rows (target {max_train_rows})")
    return kept


def _drop_index(dataset_id: int) -> None:
    if vs.hnsw_index_exists(dataset_id):
        vs.drop_hnsw_index(dataset_id)
    execute("DELETE FROM hnsw_indexes WHERE dataset_id = %s", (dataset_id,))


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def run_one(
    args: argparse.Namespace,
    metric: DistanceMetric,
    method: str,
    ef_search: int | None,
    size_label: str,
) -> dict:
    config = ExperimentConfig(
        dataset_id=args.dataset_id,
        k=args.k,
        distance_metric=metric,
        method=method,
        ef_search=ef_search,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        max_queries=args.max_queries,
        repetitions=args.repetitions,
        seed=args.seed,
        name=f"{method} k={args.k} ef={ef_search} size={size_label}",
    )
    return run_experiment(config)


def flatten(result: dict) -> dict:
    """Turn one experiment result into a flat CSV row."""
    detail = get_experiment(int(result["id"])) or {}
    environment = detail.get("environment") or {}
    if isinstance(environment, str):
        environment = json.loads(environment)

    row = {field: result.get(field) for field in FIELDS}
    row["measured_queries"] = environment.get("measured_queries")
    row["preprocessing_mode"] = environment.get("preprocessing_mode")
    row["used_sequential_scan"] = environment.get("used_sequential_scan")
    row["applied_ef_search"] = environment.get("applied_ef_search")
    for column, key in ENV_KEYS.items():
        row[column] = environment.get(key)
    return row


def print_warning_if_index_unused(row: dict) -> None:
    if row["retrieval_method"] == "hnsw" and not row["used_index_scan"]:
        print(
            "  WARNING: HNSW run did not use the index "
            f"(sequential_scan={row['used_sequential_scan']}, "
            f"applied_ef_search={row['applied_ef_search']}). "
            "The latency belongs to an exhaustive scan, not to HNSW.",
            file=sys.stderr,
        )


def main() -> int:
    args = parse_args()

    if args.repetitions < 1:
        print("--repetitions must be at least 1", file=sys.stderr)
        return 2

    dataset = fetch_one("SELECT * FROM datasets WHERE id = %s", (args.dataset_id,))
    if dataset is None:
        print(f"Dataset {args.dataset_id} not found", file=sys.stderr)
        return 1

    print(
        f"dataset {args.dataset_id}: {dataset['name']} "
        f"(preprocessing_mode={dataset.get('preprocessing_mode')}, "
        f"dimension={dataset['dimension']})"
    )
    print(
        f"k={args.k} metric={args.metric} queries={args.max_queries} "
        f"repetitions={args.repetitions} seed={args.seed}"
    )

    metric = DistanceMetric(args.metric)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    ef_values = [int(e) for e in args.ef_search.split(",") if e.strip()]

    # The dataset is mutated in place when a size sweep is requested, so the
    # original full-size training pool is snapshotted first and restored at
    # the end even if a run fails.
    original_rows: list[tuple] = snapshot_pool(args.dataset_id) if sizes else []

    rows: list[dict] = []
    experiment_ids: list[int] = []
    baseline_by_size: dict[int | None, dict] = {}

    try:
        for size in sizes or [None]:
            label = "full" if size is None else str(size)
            if size is not None:
                print(f"[size {size}] rebuilding training pool")
                restore_pool(args.dataset_id, original_rows)
                subset_dataset(args.dataset_id, size, args.seed)

            print(f"[size {label}] exact baseline")
            exact_result = run_one(args, metric, "exact", None, label)
            exact = flatten(exact_result)
            experiment_ids.append(int(exact_result["id"]))
            rows.append(exact)
            baseline_by_size[size] = exact

            for ef in ef_values:
                print(f"[size {label}] HNSW ef_search={ef}")
                approx_result = run_one(args, metric, "hnsw", ef, label)
                approx = flatten(approx_result)
                experiment_ids.append(int(approx_result["id"]))
                approx["speedup_p50"] = _ratio(
                    exact["latency_p50_ms"], approx["latency_p50_ms"]
                )
                approx["speedup_avg"] = _ratio(
                    exact["latency_avg_ms"], approx["latency_avg_ms"]
                )
                rows.append(approx)
                print_warning_if_index_unused(approx)
    finally:
        if sizes and original_rows:
            print("restoring original training pool")
            restore_pool(args.dataset_id, original_rows)

    _print_table(rows)

    if args.output:
        _write_csv(rows, args.output)
        print(f"CSV written to {args.output}")
    if args.json_output:
        Path(args.json_output).write_text(
            json.dumps(rows, indent=2, default=str), encoding="utf-8"
        )
        print(f"JSON written to {args.json_output}")
    if args.repetition_details:
        _write_repetition_details(experiment_ids, args.repetition_details)
        print(f"Per-repetition CSV written to {args.repetition_details}")
    if args.repetition_summary:
        _write_repetition_summary(experiment_ids, args.repetition_summary)
        print(f"Repetition summary CSV written to {args.repetition_summary}")

    return 0


def _ratio(baseline: float | None, measured: float | None) -> float | None:
    if not baseline or not measured:
        return None
    return baseline / measured


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'size':>8} {'method':>8} {'k':>3} {'ef':>5} {'reps':>4} "
        f"{'recall':>8} {'acc':>8} {'macroF1':>8} "
        f"{'p50ms':>9} {'p99ms':>9} {'buildms':>9} {'idx':>5} {'x':>7}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        # ASCII only: the Windows console codepage mangles an em dash.
        def fmt(value: float | None, digits: int = 4) -> str:
            return "-" if value is None else f"{value:.{digits}f}"

        print(
            f"{str(row['train_size']):>8} {row['retrieval_method']:>8} "
            f"{row['k']:>3} {str(row['hnsw_ef_search'] or '-'):>5} "
            f"{row['repetitions']:>4} "
            f"{fmt(row['recall_at_k']):>8} {fmt(row['accuracy']):>8} "
            f"{fmt(row['macro_f1']):>8} "
            f"{fmt(row['latency_p50_ms'], 3):>9} "
            f"{fmt(row['latency_p99_ms'], 3):>9} "
            f"{fmt(row['index_build_ms'], 1):>9} "
            f"{str(row['used_index_scan']):>5} "
            f"{fmt(row['speedup_p50'], 1):>7}"
        )


def _write_csv(rows: list[dict], path: str, fieldnames: list[str] | None = None) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames or FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_repetition_details(experiment_ids: list[int], path: str) -> None:
    """One row per (experiment, repetition), read back from the database.

    The values come from ``experiment_repetitions``, which is written by the
    same service the API uses, so this file cannot drift from the stored run.
    """
    rows: list[dict] = []
    for experiment_id in experiment_ids:
        experiment = get_experiment(experiment_id) or {}
        for rep in get_repetitions(experiment_id):
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "dataset_id": experiment.get("dataset_id"),
                    "train_size": experiment.get("train_size"),
                    "retrieval_method": experiment.get("retrieval_method"),
                    "ef_search": experiment.get("hnsw_ef_search"),
                    "repetition": rep["repetition"],
                    "unique_queries": rep["unique_queries"],
                    "samples": rep["samples"],
                    "latency_mean_ms": rep["latency_avg_ms"],
                    "latency_p50_ms": rep["latency_p50_ms"],
                    "latency_p95_ms": rep["latency_p95_ms"],
                    "latency_p99_ms": rep["latency_p99_ms"],
                    "recall_at_k": rep["recall_at_k"],
                    "tie_aware_recall_at_k": rep["tie_aware_recall_at_k"],
                    "accuracy": rep["accuracy"],
                    "precision_macro": rep["precision_macro"],
                    "recall_macro": rep["recall_macro"],
                    "macro_f1": rep["macro_f1"],
                }
            )
    _write_csv(rows, path, REPETITION_FIELDS)


_SUMMARY_METRICS = [
    ("latency_mean_ms", "latency_avg_ms"),
    ("latency_p50_ms", "latency_p50_ms"),
    ("latency_p95_ms", "latency_p95_ms"),
    ("latency_p99_ms", "latency_p99_ms"),
    ("recall_at_k", "recall_at_k"),
    ("tie_aware_recall_at_k", "tie_aware_recall_at_k"),
    ("accuracy", "accuracy"),
    ("precision_macro", "precision_macro"),
    ("recall_macro", "recall_macro"),
    ("macro_f1", "macro_f1"),
]


def _write_repetition_summary(experiment_ids: list[int], path: str) -> None:
    """Median and dispersion of each metric across repetitions.

    The repetitions share one query set and one index, so this spread is
    run-to-run variability, not sampling error over a population.
    """
    rows: list[dict] = []
    for experiment_id in experiment_ids:
        experiment = get_experiment(experiment_id) or {}
        reps = get_repetitions(experiment_id)
        if not reps:
            continue
        for output_name, column in _SUMMARY_METRICS:
            summary = repetition_summary([rep[column] for rep in reps])
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "dataset_id": experiment.get("dataset_id"),
                    "train_size": experiment.get("train_size"),
                    "retrieval_method": experiment.get("retrieval_method"),
                    "ef_search": experiment.get("hnsw_ef_search"),
                    "repetitions": len(reps),
                    "metric": output_name,
                    **summary,
                }
            )
    _write_csv(rows, path, SUMMARY_FIELDS)


if __name__ == "__main__":
    raise SystemExit(main())
