"""Batch experiment runner.

Runs the full experimental matrix described in the project proposal:

* an **exact baseline** for every training-set size, and
* an **HNSW sweep** over a list of ``ef_search`` values for every size.

Results are printed as a table and can be exported as CSV or JSON, which is
what the final report's charts are built from.

Usage::

    python -m scripts.run_experiments --dataset-id 1 --sizes 10000,50000 --ef-search 10,50,100
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
from app.services.experiments import ExperimentConfig, run_experiment  # noqa: E402
from app.services.metrics import DistanceMetric  # noqa: E402
from app.services import vector_store as vs  # noqa: E402

FIELDS = [
    "dataset_id",
    "train_size",
    "test_size",
    "retrieval_method",
    "k",
    "distance_metric",
    "hnsw_ef_search",
    "recall_at_k",
    "accuracy",
    "macro_f1",
    "latency_avg_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "index_build_ms",
    "used_index_scan",
    "total_runtime_s",
]


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
    parser.add_argument(
        "--metric", choices=["l2", "cosine", "l1"], default="l2"
    )
    parser.add_argument("--max-queries", type=int, default=1000)
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--output", default="", help="CSV output path")
    parser.add_argument("--json", dest="json_output", default="", help="JSON output path")
    return parser.parse_args()


def subset_dataset(dataset_id: int, max_train_rows: int | None) -> None:
    """Reduce the training pool of a dataset to ``max_train_rows`` rows.

    The rows are removed from the database (not just ignored) so that the
    measured latency reflects the actual table size.
    """
    if max_train_rows is None:
        return

    with vs.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                vs.sql.SQL("SELECT count(*) AS c FROM {t}").format(
                    t=vs.sql.Identifier(vs.train_table(dataset_id))
                )
            )
            total = int(cur.fetchone()["c"])
            if total <= max_train_rows:
                return

            cur.execute(
                vs.sql.SQL(
                    "DELETE FROM {t} WHERE id NOT IN ("
                    "  SELECT id FROM {t} ORDER BY id LIMIT {n}"
                    ")"
                ).format(
                    t=vs.sql.Identifier(vs.train_table(dataset_id)),
                    n=vs.sql.Literal(max_train_rows),
                )
            )
            removed = total - max_train_rows
    print(f"  reduced training pool from {total} to {max_train_rows} rows")


def main() -> int:
    args = parse_args()

    dataset = fetch_one("SELECT * FROM datasets WHERE id = %s", (args.dataset_id,))
    if dataset is None:
        print(f"Dataset {args.dataset_id} not found", file=sys.stderr)
        return 1

    metric = DistanceMetric(args.metric)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    ef_values = [int(e) for e in args.ef_search.split(",") if e.strip()]

    # The dataset is mutated in place when a size sweep is requested, so the
    # original full-size training pool is snapshotted first.
    original_rows: list[tuple] = []
    if sizes:
        with vs.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    vs.sql.SQL(
                        "SELECT id, row_index, label, embedding::text FROM {t} ORDER BY id"
                    ).format(t=vs.sql.Identifier(vs.train_table(args.dataset_id)))
                )
                original_rows = list(cur.fetchall())

    rows: list[dict] = []

    try:
        # --- Exact baseline per size -------------------------------------
        for size in sizes or [None]:
            if size is not None:
                print(f"[size {size}] rebuilding training pool")
                restore_pool(args.dataset_id, original_rows)
                subset_dataset(args.dataset_id, size)

            print(f"[size {size}] exact baseline")
            result = run_experiment(
                ExperimentConfig(
                    dataset_id=args.dataset_id,
                    k=args.k,
                    distance_metric=metric,
                    method="exact",
                    max_queries=args.max_queries,
                    name=f"exact k={args.k} size={size or 'full'}",
                )
            )
            rows.append(_row(result))

        # --- HNSW sweep per size -----------------------------------------
        for size in sizes or [None]:
            if size is not None:
                print(f"[size {size}] rebuilding training pool")
                restore_pool(args.dataset_id, original_rows)
                subset_dataset(args.dataset_id, size)

            for ef in ef_values:
                print(f"[size {size}] HNSW ef_search={ef}")
                result = run_experiment(
                    ExperimentConfig(
                        dataset_id=args.dataset_id,
                        k=args.k,
                        distance_metric=metric,
                        method="hnsw",
                        ef_search=ef,
                        hnsw_m=args.hnsw_m,
                        hnsw_ef_construction=args.hnsw_ef_construction,
                        max_queries=args.max_queries,
                        name=f"hnsw ef={ef} k={args.k} size={size or 'full'}",
                    )
                )
                rows.append(_row(result))
    finally:
        if sizes and original_rows:
            print("restoring original training pool")
            restore_pool(args.dataset_id, original_rows)

    _print_table(rows)

    if args.output:
        _write_csv(rows, args.output)
        print(f"CSV written to {args.output}")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"JSON written to {args.json_output}")

    return 0


def restore_pool(dataset_id: int, original_rows: list[tuple]) -> None:
    """Restore the full training pool from a snapshot."""
    table = vs.train_table(dataset_id)
    with vs.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"TRUNCATE TABLE {table}")
            if not original_rows:
                return
            with cur.copy(
                vs.sql.SQL(
                    "COPY {t} (id, row_index, label, embedding) FROM STDIN"
                ).format(t=vs.sql.Identifier(table))
            ) as copy:
                for row in original_rows:
                    copy.write_row(
                        (int(row["id"]), int(row["row_index"]), int(row["label"]), row["embedding"])
                    )
    vs.drop_hnsw_index(dataset_id)
    execute("DELETE FROM hnsw_indexes WHERE dataset_id = %s", (dataset_id,))


def _row(result: dict) -> dict:
    return {field: result.get(field) for field in FIELDS}


def _print_table(rows: list[dict]) -> None:
    header = (
        f"{'size':>8} {'method':>8} {'k':>3} {'ef':>5} "
        f"{'recall':>8} {'acc':>8} {'macroF1':>8} {'p50ms':>9} {'p95ms':>9} {'buildms':>9}"
    )
    print()
    print(header)
    print("-" * len(header))
    for row in rows:
        def fmt(value: float | None, digits: int = 4) -> str:
            return "—" if value is None else f"{value:.{digits}f}"

        print(
            f"{row['train_size']:>8} {row['retrieval_method']:>8} {row['k']:>3} "
            f"{str(row['hnsw_ef_search'] or '—'):>5} "
            f"{fmt(row['recall_at_k']):>8} {fmt(row['accuracy']):>8} "
            f"{fmt(row['macro_f1']):>8} {fmt(row['latency_p50_ms'], 3):>9} "
            f"{fmt(row['latency_p95_ms'], 3):>9} {fmt(row['index_build_ms'], 1):>9}"
        )


def _write_csv(rows: list[dict], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
