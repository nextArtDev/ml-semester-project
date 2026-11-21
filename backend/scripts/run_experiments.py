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
* Each requested pool size is materialised as a derived dataset with its own
  preprocessing state, fitted on that size's training rows only. Within-class
  selection uses ``random.Random(f"{seed}:{label}")``, so any integer seed is
  reproducible without PostgreSQL's restricted ``setseed``.
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
import random
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.database import execute, fetch_one, transaction  # noqa: E402
from app.services import vector_store as vs  # noqa: E402
from app.services.benchmark_order import balanced_rounds  # noqa: E402
from app.services.classification import repetition_summary  # noqa: E402
from app.services.dataset_stats import dataset_fingerprint  # noqa: E402
from app.services.reproducibility import (  # noqa: E402
    preprocessing_state_hash,
    selected_ids_provenance,
    variant_identity_hash,
)
from app.services.preprocessing import (  # noqa: E402
    PreprocessState,
    fit_preprocessor,
    inverse_transform_vectors,
    labels_to_original,
    transform,
    vector_literal,
)
from app.services.experiments import (  # noqa: E402
    ExperimentConfig,
    get_experiment,
    get_repetitions,
    run_experiment,
)
from app.services.metrics import DistanceMetric  # noqa: E402

FIELDS = [
    "experiment_id",
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
    "query_sampling",
    "query_sample_seed",
    "benchmark_round",
    "benchmark_position",
    "benchmark_seed",
    "run_id",
    "warm_cache",
    "total_experiment_runtime_s",
    "index_preparation_s",
    "index_build_s",
    "ground_truth_computation_s",
    "ground_truth_cache_hit",
    "ground_truth_lookup_s",
    "retrieval_total_s",
    "summed_query_execution_time_s",
    "retrieval_batch_wall_time_s",
    "retrieval_mean_per_query_ms",
    "metrics_computation_s",
    "persistence_s",
    "orchestration_overhead_s",
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
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1000,
        help="Queries to evaluate; 0 or omitted upper bound means all test rows.",
    )
    parser.add_argument(
        "--query-sampling",
        choices=["first", "stratified"],
        default="stratified",
        help="Select the first ids or a seeded class-stratified sample.",
    )
    parser.add_argument(
        "--query-sample-seed",
        type=int,
        default=0,
        help="Seed for stratified query selection.",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=3,
        help="Number of paired benchmark rounds; each round runs every configuration once. "
        "Three or more guarantee every configuration is both early and late.",
    )
    parser.add_argument(
        "--order-seed",
        type=int,
        default=0,
        help="Seed for the balanced per-round execution order.",
    )
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
    parser.add_argument(
        "--query-samples",
        default="",
        help="JSON output path mapping each experiment to its evaluated query ids.",
    )
    parser.add_argument(
        "--benchmark-orders",
        default="",
        help="JSON output path with the complete order and seed of every round.",
    )
    parser.add_argument(
        "--variant-ids",
        default="",
        help="JSON output path with exact selected ids, stored once per derived variant.",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Label persisted on every experiment so artifacts can be tied to this run.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Training-pool management
# ---------------------------------------------------------------------------
def _snapshot_pool(dataset_id: int, which: str) -> list[tuple]:
    table = vs.train_table(dataset_id) if which == "train" else vs.test_table(dataset_id)
    with vs.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                vs.sql.SQL(
                    "SELECT id, row_index, label, embedding::text FROM {t} ORDER BY id"
                ).format(t=vs.sql.Identifier(table))
            )
            return list(cur.fetchall())


def snapshot_pool(dataset_id: int) -> list[tuple]:
    """Snapshot the original training pool (legacy in-place helper)."""
    return _snapshot_pool(dataset_id, "train")


def snapshot_test_pool(dataset_id: int) -> list[tuple]:
    """Snapshot the held-out query pool."""
    return _snapshot_pool(dataset_id, "test")


def legacy_restore_pool(dataset_id: int, rows: list[tuple]) -> None:
    """Destructively restore a training pool. Use only on disposable datasets."""
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


def restore_pool(dataset_id: int, rows: list[tuple], *, allow_destructive: bool = False) -> None:
    """Deprecated wrapper for :func:`legacy_restore_pool`.

    The normal size-sweep path never restores a pool in place. An explicit
    ``allow_destructive=True`` is required because this function TRUNCATEs a
    vector table.
    """
    if not allow_destructive:
        raise RuntimeError(
            "restore_pool is destructive; pass allow_destructive=True only for "
            "a disposable test dataset."
        )
    legacy_restore_pool(dataset_id, rows)


def _choose_subset_ids(rows: list[tuple], max_train_rows: int, seed: int) -> list[int]:
    """Choose rows for a pool size using exact largest-remainder quotas."""
    total = len(rows)
    if total <= max_train_rows:
        return [int(row["id"]) for row in rows]

    by_label: dict[int, list[int]] = {}
    for row in rows:
        by_label.setdefault(int(row["label"]), []).append(int(row["id"]))
    counts = {label: len(ids) for label, ids in by_label.items()}
    allocation = vs.largest_remainder_allocation(counts, max_train_rows)

    keep: list[int] = []
    for label in sorted(by_label):
        ids = by_label[label]
        need = allocation[label]
        if need >= len(ids):
            keep.extend(ids)
        elif need > 0:
            keep.extend(sorted(random.Random(f"{seed}:{label}").sample(ids, need)))
    return sorted(keep)


def legacy_subset_dataset(dataset_id: int, max_train_rows: int | None, seed: int) -> int | None:
    """Reduce the training pool to exactly ``max_train_rows`` rows.

    Rows are removed physically so the measured latency reflects the real table
    size.  Class quotas use a largest-remainder allocation, so the resulting
    count equals the requested target whenever the pool is large enough and
    every class is represented when the target permits it.  Within each class
    the kept ids are chosen by ``random.Random(f"{seed}:{label}")``, which is
    deterministic for any integer seed and avoids PostgreSQL's ``setseed``
    range restriction entirely.

    Returns the actual resulting row count, or ``None`` when no reduction was
    requested.
    """
    if max_train_rows is None:
        return None

    table = vs.train_table(dataset_id)
    rows = _snapshot_pool(dataset_id, "train")
    total = len(rows)
    if total <= max_train_rows:
        return total

    keep = _choose_subset_ids(rows, max_train_rows, seed)
    with vs.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                vs.sql.SQL("DELETE FROM {t} WHERE NOT (id = ANY(%s))").format(
                    t=vs.sql.Identifier(table)
                ),
                (keep,),
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


def subset_dataset(
    dataset_id: int,
    max_train_rows: int | None,
    seed: int,
    *,
    allow_destructive: bool = False,
) -> int | None:
    """Deprecated wrapper for :func:`legacy_subset_dataset`.

    The normal size-sweep path never deletes rows from an original pool.
    """
    if not allow_destructive:
        raise RuntimeError(
            "subset_dataset is destructive; pass allow_destructive=True only "
            "for a disposable test dataset."
        )
    return legacy_subset_dataset(dataset_id, max_train_rows, seed)


def _drop_index(dataset_id: int) -> None:
    if vs.hnsw_index_exists(dataset_id):
        vs.drop_hnsw_index(dataset_id)
    execute("DELETE FROM hnsw_indexes WHERE dataset_id = %s", (dataset_id,))


def _insert_variant_dataset_row(
    cur, parent: dict, state: PreprocessState, name: str, actual_size: int
) -> int:
    """Insert the derived dataset row using the caller's transaction cursor."""
    cur.execute(
        """
        INSERT INTO datasets
            (name, source, original_name, n_rows, n_features, n_classes,
             dimension, class_labels, preprocessing, is_seeded,
             preprocessing_mode)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
        RETURNING id
        """,
        (
            name,
            "derived",
            parent.get("original_name") or parent["name"],
            actual_size,
            len(state.output_columns),
            len(state.class_labels),
            state.dimension,
            json.dumps(state.class_labels),
            state.to_json(),
            False,
            state.mode,
        ),
    )
    return int(cur.fetchone()["id"])


def _create_variant_table(cur, dataset_id: int, which: str, dimension: int) -> None:
    """Create one typed vector table using the caller's transaction cursor."""
    table = vs.train_table(dataset_id) if which == "train" else vs.test_table(dataset_id)
    cur.execute(
        vs.sql.SQL(
            "CREATE TABLE {t} ("
            "  id          BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,"
            "  row_index   INTEGER NOT NULL,"
            "  label       SMALLINT NOT NULL,"
            "  embedding   vector({dim}) NOT NULL"
            ")"
        ).format(t=vs.sql.Identifier(table), dim=vs.sql.Literal(int(dimension)))
    )
    cur.execute(
        vs.sql.SQL("CREATE INDEX {i} ON {t} (row_index)").format(
            i=vs.sql.Identifier(f"{table}_row_idx"), t=vs.sql.Identifier(table)
        )
    )


def _create_variant_tables(cur, dataset_id: int, dimension: int) -> None:
    """Create both variant tables in one transaction."""
    _create_variant_table(cur, dataset_id, "train", dimension)
    _create_variant_table(cur, dataset_id, "test", dimension)


def _copy_variant_vectors(
    cur,
    dataset_id: int,
    which: str,
    embeddings,
    labels,
    row_indices,
) -> None:
    """COPY vectors using the caller's transaction cursor."""
    table = vs.train_table(dataset_id) if which == "train" else vs.test_table(dataset_id)
    with cur.copy(
        vs.sql.SQL("COPY {t} (row_index, label, embedding) FROM STDIN").format(
            t=vs.sql.Identifier(table)
        )
    ) as copy:
        for embedding, label, row_index in zip(embeddings, labels, row_indices):
            copy.write_row((int(row_index), int(label), embedding))


def build_preprocessed_variant(
    parent_id: int,
    max_train_rows: int,
    seed: int,
) -> tuple[dict, list[int]]:
    """Materialise a pool with preprocessing fitted on that pool alone.

    Registration, table creation and COPY share one real PostgreSQL
    transaction. Any failure rolls them back together. Exact selected ids are
    returned separately so callers can export them once without embedding tens
    of thousands of ids in every experiment configuration.
    """
    parent = fetch_one("SELECT * FROM datasets WHERE id = %s", (parent_id,))
    if parent is None:
        raise ValueError(f"Dataset {parent_id} not found")

    parent_fingerprint = dataset_fingerprint(parent_id)
    base_state = PreprocessState.from_dict(parent["preprocessing"])
    train_rows = _snapshot_pool(parent_id, "train")
    test_rows = _snapshot_pool(parent_id, "test")
    selected_ids = sorted(_choose_subset_ids(train_rows, max_train_rows, seed))
    selected_id_set = set(selected_ids)
    selected_train = [row for row in train_rows if int(row["id"]) in selected_id_set]

    raw_train = inverse_transform_vectors(
        [row["embedding"] for row in selected_train], base_state
    )
    raw_train[base_state.target_column] = labels_to_original(
        [int(row["label"]) for row in selected_train], base_state
    )

    raw_test = inverse_transform_vectors(
        [row["embedding"] for row in test_rows], base_state
    )
    raw_test[base_state.target_column] = labels_to_original(
        [int(row["label"]) for row in test_rows], base_state
    )

    state, preprocessor = fit_preprocessor(
        raw_train, base_state.target_column, base_state.mode
    )
    if state.class_labels != base_state.class_labels:
        raise ValueError(
            "The subsampled pool changed the label mapping; refusing to create "
            "a variant whose encoded labels would differ from the parent dataset."
        )
    if (
        state.numeric_columns != base_state.numeric_columns
        or state.binary_columns != base_state.binary_columns
        or state.categorical_columns != base_state.categorical_columns
    ):
        raise ValueError(
            "Column-type classification changed after subsampling; refusing to "
            "create a variant with silently different feature geometry."
        )

    train_matrix, train_labels = transform(raw_train, state, preprocessor)
    test_matrix, test_labels = transform(raw_test, state, preprocessor)

    ids_provenance = selected_ids_provenance(selected_ids)
    state_hash = preprocessing_state_hash(state)
    identity_input = {
        "parent_fingerprint": parent_fingerprint,
        "target_size": max_train_rows,
        "training_pool_seed": seed,
        "selected_id_count": ids_provenance["count"],
        "selected_id_sha256": ids_provenance["sha256"],
        "preprocessing_mode": state.mode,
        "preprocessing_state_sha256": state_hash,
    }
    identity_hash = variant_identity_hash(identity_input)
    name = f"{parent['name']} [pool={len(selected_train)}, seed={seed}]"
    dimension = int(state.dimension)
    train_vectors = vector_literal(train_matrix)
    test_vectors = vector_literal(test_matrix)
    train_row_indices = [int(row["row_index"]) for row in selected_train]
    test_row_indices = [int(row["row_index"]) for row in test_rows]

    with transaction() as conn:
        with conn.cursor() as cur:
            variant_id = _insert_variant_dataset_row(
                cur, parent, state, name, len(selected_train)
            )
            _create_variant_tables(cur, variant_id, dimension)
            _copy_variant_vectors(
                cur, variant_id, "train", train_vectors, train_labels, train_row_indices
            )
            _copy_variant_vectors(
                cur, variant_id, "test", test_vectors, test_labels, test_row_indices
            )

    metadata = {
        **identity_input,
        "dataset_id": variant_id,
        "parent_dataset_id": parent_id,
        "actual_size": len(selected_train),
        "variant_identity_sha256": identity_hash,
        "derived_fingerprint": dataset_fingerprint(variant_id),
        "state_source": "refit_on_variant_training_pool",
        "variant_policy": "immutable_retained",
    }
    return metadata, selected_ids


# ---------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------
def _config_key(method: str, ef_search: int | None) -> str:
    return f"{method}:{ef_search if ef_search is not None else '-'}"


def _parse_key(key: str) -> tuple[str, int | None]:
    method, raw_ef = key.split(":", 1)
    return method, None if raw_ef == "-" else int(raw_ef)


def run_one(
    args: argparse.Namespace,
    metric: DistanceMetric,
    method: str,
    ef_search: int | None,
    size_label: str,
    benchmark_round: int,
    benchmark_position: int,
    benchmark_order: str,
    dataset_id: int | None = None,
    extra: dict | None = None,
) -> dict:
    config = ExperimentConfig(
        dataset_id=args.dataset_id if dataset_id is None else dataset_id,
        k=args.k,
        distance_metric=metric,
        method=method,
        ef_search=ef_search,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        max_queries=args.max_queries if args.max_queries > 0 else None,
        repetitions=args.repetitions,
        seed=args.seed,
        query_sampling=args.query_sampling,
        query_sample_seed=args.query_sample_seed,
        benchmark_round=benchmark_round,
        benchmark_position=benchmark_position,
        benchmark_seed=args.order_seed,
        benchmark_order=benchmark_order,
        run_id=args.run_id or None,
        extra=extra or {},
        name=(
            f"{method} k={args.k} ef={ef_search} size={size_label} "
            f"round={benchmark_round} pos={benchmark_position}"
        ),
    )
    return run_experiment(config)


def flatten(result: dict) -> dict:
    """Turn one experiment result into a flat CSV row."""
    detail = get_experiment(int(result["id"])) or {}
    environment = detail.get("environment") or {}
    if isinstance(environment, str):
        environment = json.loads(environment)

    row = {field: result.get(field) for field in FIELDS}
    row["experiment_id"] = int(result["id"])
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
    if args.rounds < 1:
        print("--rounds must be at least 1", file=sys.stderr)
        return 2

    dataset = fetch_one("SELECT * FROM datasets WHERE id = %s", (args.dataset_id,))
    if dataset is None:
        print(f"Dataset {args.dataset_id} not found", file=sys.stderr)
        return 1

    metric = DistanceMetric(args.metric)
    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    ef_values = [int(e) for e in args.ef_search.split(",") if e.strip()]
    configs = [_config_key("exact", None)] + [
        _config_key("hnsw", ef) for ef in ef_values
    ]
    rounds = balanced_rounds(configs, args.rounds, args.order_seed)

    print(
        f"dataset {args.dataset_id}: {dataset['name']} "
        f"(preprocessing_mode={dataset.get('preprocessing_mode')}, "
        f"dimension={dataset['dimension']})"
    )
    print(
        f"k={args.k} metric={args.metric} queries={args.max_queries} "
        f"repetitions={args.repetitions} seed={args.seed}"
    )
    print(
        f"rounds={args.rounds} order_seed={args.order_seed} "
        f"query_sampling={args.query_sampling} query_sample_seed={args.query_sample_seed}"
    )
    print("warm buffer cache; no cold-cache procedure was executed")

    # Each requested size owns a derived dataset row with its own fitted
    # preprocessing state. The original dataset and its vector tables are
    # never mutated.
    print("original dataset and vector tables are left unchanged")

    rows: list[dict] = []
    experiment_ids: list[int] = []
    query_samples: dict[str, dict] = {}
    benchmark_orders: dict[str, dict] = {}
    variant_artifacts: dict[str, dict] = {}

    for size in sizes or [None]:
        label = "full" if size is None else str(size)
        variant_meta: dict | None = None
        if size is not None:
            variant_meta, selected_ids = build_preprocessed_variant(
                args.dataset_id, size, args.seed
            )
            effective_dataset_id = int(variant_meta["dataset_id"])
            actual_size = int(variant_meta["actual_size"])
            variant_artifacts[str(effective_dataset_id)] = {
                **variant_meta,
                "selected_ids": selected_ids,
                "selected_ids_canonical": (
                    "sorted integer ids; SHA-256 over canonical JSON array"
                ),
            }
            print(
                f"[size {label}] derived dataset {effective_dataset_id} "
                f"with {actual_size} rows and state-specific query vectors"
            )
        else:
            effective_dataset_id = args.dataset_id
            actual_size = vs.count_vectors(effective_dataset_id, "train")

        extra = {"pool_variant": variant_meta} if variant_meta is not None else {}

        # The index is built on the first HNSW run of this size and reused
        # for every later configuration and round. It is not rebuilt
        # between ef_search values: only ef_search changes.
        for round_index, order in enumerate(rounds, start=1):
            print(f"[size {label}] round {round_index}: {' -> '.join(order)}")
            order_json = json.dumps(order)
            round_key = f"{label}:{round_index}"
            benchmark_orders[round_key] = {
                "dataset_id": effective_dataset_id,
                "parent_dataset_id": args.dataset_id,
                "train_size_target": size,
                "train_size_actual": actual_size,
                "preprocessing_state_source": (
                    variant_meta["state_source"]
                    if variant_meta is not None
                    else "dataset_row"
                ),
                "parent_fingerprint": (
                    variant_meta["parent_fingerprint"]
                    if variant_meta is not None
                    else None
                ),
                "derived_fingerprint": (
                    variant_meta["derived_fingerprint"]
                    if variant_meta is not None
                    else None
                ),
                "selected_id_count": (
                    variant_meta["selected_id_count"]
                    if variant_meta is not None
                    else None
                ),
                "selected_id_sha256": (
                    variant_meta["selected_id_sha256"]
                    if variant_meta is not None
                    else None
                ),
                "preprocessing_state_sha256": (
                    variant_meta["preprocessing_state_sha256"]
                    if variant_meta is not None
                    else None
                ),
                "variant_identity_sha256": (
                    variant_meta["variant_identity_sha256"]
                    if variant_meta is not None
                    else None
                ),
                "benchmark_round": round_index,
                "order_seed": args.order_seed,
                "training_pool_seed": args.seed,
                "query_sample_seed": args.query_sample_seed,
                "run_id": args.run_id or None,
                "order": list(order),
                "experiment_ids": [],
            }
            for position, key in enumerate(order, start=1):
                method, ef = _parse_key(key)
                print(f"[size {label}] round {round_index} pos {position}: {method} ef={ef}")
                result = run_one(
                    args,
                    metric,
                    method,
                    ef,
                    label,
                    round_index,
                    position,
                    order_json,
                    dataset_id=effective_dataset_id,
                    extra=extra,
                )
                experiment_id = int(result["id"])
                benchmark_orders[round_key]["experiment_ids"].append(experiment_id)
                flat = flatten(result)
                experiment_ids.append(experiment_id)
                rows.append(flat)
                if method == "hnsw":
                    print_warning_if_index_unused(flat)

                detail = get_experiment(experiment_id) or {}
                query_samples[str(experiment_id)] = {
                    "dataset_id": effective_dataset_id,
                    "parent_dataset_id": args.dataset_id,
                    "train_size": flat["train_size"],
                    "benchmark_round": flat["benchmark_round"],
                    "benchmark_position": flat["benchmark_position"],
                    "query_sampling": flat["query_sampling"],
                    "query_sample_seed": flat["query_sample_seed"],
                    "query_ids": detail.get("evaluated_query_ids"),
                }

    _add_speedups(rows)
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
    if args.query_samples:
        Path(args.query_samples).write_text(
            json.dumps(query_samples, indent=2), encoding="utf-8"
        )
        print(f"Query ids written to {args.query_samples}")
    if args.benchmark_orders:
        Path(args.benchmark_orders).write_text(
            json.dumps(benchmark_orders, indent=2), encoding="utf-8"
        )
        print(f"Benchmark orders written to {args.benchmark_orders}")
    if args.variant_ids:
        Path(args.variant_ids).write_text(
            json.dumps(
                {
                    "algorithm": {
                        "canonical_json": "sort_keys=True, separators=(',', ':'), ensure_ascii=False",
                        "selected_ids": "sorted ascending integer database ids",
                        "hash": "sha256 hex over UTF-8 canonical JSON",
                    },
                    "variants": variant_artifacts,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"Variant ids written to {args.variant_ids}")

    return 0


def _add_speedups(rows: list[dict]) -> None:
    """Attach speed-ups from the mean exact baseline of the same pool size.

    The baseline is averaged over all benchmark rounds instead of taken from a
    single run, so a configuration's speed-up does not depend on whether an
    exact run happened to be scheduled immediately before it.
    """
    baselines: dict[object, dict[str, float]] = {}
    for row in rows:
        if row["retrieval_method"] != "exact":
            continue
        size = row["train_size"]
        bucket = baselines.setdefault(size, {"p50": [], "avg": []})
        for key, source in (("p50", "latency_p50_ms"), ("avg", "latency_avg_ms")):
            value = row.get(source)
            if value not in (None, ""):
                bucket[key].append(float(value))

    for row in rows:
        if row["retrieval_method"] != "hnsw":
            continue
        bucket = baselines.get(row["train_size"])
        if not bucket:
            continue
        for key, source, target in (
            ("p50", "latency_p50_ms", "speedup_p50"),
            ("avg", "latency_avg_ms", "speedup_avg"),
        ):
            values = bucket[key]
            baseline = sum(values) / len(values) if values else None
            row[target] = _ratio(baseline, row.get(source))


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
