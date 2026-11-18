"""Experiment orchestration.

An *experiment* is one retrieval configuration evaluated against one
dataset. The service:

1. builds (or reuses) the HNSW index,
2. runs the exact baseline to obtain ground-truth neighbours,
3. runs the configured retrieval method, repeating the timed queries to
   reduce measurement noise,
4. classifies both, compares accuracy / macro-F1 / recall@K / latency,
5. persists everything to the ``experiments`` and ``query_latencies`` tables
   together with the environment the run happened in.

Isolation guarantees
--------------------
The exact and the approximate arm of a comparison must differ *only* in the
retrieval method. The service therefore enforces, within a single run, that
the two arms use:

* the same training pool (the same ``fv_<id>`` table),
* the same query vectors, in the same order,
* the same fitted preprocessing state,
* the same distance metric,
* the same value of k,
* the same majority-voting implementation.

The ground truth is always recomputed with the exhaustive query, so Recall@K
is always a like-for-like comparison.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Sequence

from app.core.database import connection, execute, fetch_all, fetch_one
from app.core.environment import capture_environment
from app.services import vector_store as vs
from app.services.classification import (
    ClassificationMetrics,
    evaluate_predictions,
    latency_stats,
    majority_vote,
    recall_at_k,
    repetition_summary,
    tie_aware_recall_at_k,
)
from app.services.dataset_stats import dataset_fingerprint
from app.services.metrics import DistanceMetric
from app.services.preprocessing import PreprocessState
from app.services.vector_store import TieTolerance

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run.

    ``repetitions`` is the number of *complete* timed passes over the same
    query set. It is not a multiplier applied to the recorded query count:
    every pass really executes and every individual latency is persisted.
    """

    dataset_id: int
    k: int = 5
    distance_metric: DistanceMetric = DistanceMetric.L2
    method: str = "exact"  # 'exact' | 'hnsw'
    ef_search: int | None = None
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    max_queries: int | None = 2_000
    name: str | None = None
    weighted_voting: bool = False
    use_operator: bool = True
    repetitions: int = 1
    seed: int = 0
    force_rebuild_index: bool = False
    tie_tolerance_relative: float = 1e-6
    tie_tolerance_absolute: float = 1e-6
    query_sampling: str = "first"
    query_sample_seed: int = 0
    benchmark_round: int | None = None
    benchmark_position: int | None = None
    benchmark_seed: int | None = None
    benchmark_order: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.method not in {"exact", "hnsw"}:
            raise ValueError(
                f"method must be 'exact' or 'hnsw' (got {self.method!r})"
            )
        if self.k < 1:
            raise ValueError(f"k must be at least 1 (got {self.k})")
        if self.repetitions < 1:
            raise ValueError(
                f"repetitions must be at least 1 (got {self.repetitions})"
            )
        if self.method == "hnsw" and self.ef_search is None:
            raise ValueError("ef_search is required for the HNSW method")
        if self.method == "hnsw" and self.ef_search < 1:
            raise ValueError(f"ef_search must be at least 1 (got {self.ef_search})")
        if not 0.0 <= self.tie_tolerance_relative < 1.0:
            raise ValueError("tie_tolerance_relative must be in [0, 1)")
        if self.tie_tolerance_absolute < 0.0:
            raise ValueError("tie_tolerance_absolute must be non-negative")
        if self.query_sampling not in {"first", "stratified"}:
            raise ValueError("query_sampling must be 'first' or 'stratified'")
        if self.benchmark_round is not None and self.benchmark_round < 1:
            raise ValueError("benchmark_round must be at least 1")
        if self.benchmark_position is not None and self.benchmark_position < 1:
            raise ValueError("benchmark_position must be at least 1")


def _load_test_vectors(dataset_id: int, limit: int | None) -> list[tuple[int, int, str]]:
    return vs.load_test_vectors(dataset_id, limit)


def _class_distribution(labels: dict[int, int] | Sequence[int]) -> dict[str, int]:
    if isinstance(labels, dict):
        counts = {int(label): int(count) for label, count in labels.items()}
    else:
        counts = {}
        for label in labels:
            counts[int(label)] = counts.get(int(label), 0) + 1
    return {str(label): counts[label] for label in sorted(counts)}


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Execute an experiment and persist its results.

    Measurement contract:

    * ``unique_queries`` queries are evaluated, chosen once from the held-out
      pool (``total_test_pool`` is the size of that pool, not the number of
      queries evaluated).
    * Every query is executed ``repetitions`` times, so the persisted latency
      row count is exactly ``repetitions * unique_queries``. The warm-up query
      is executed and discarded before the first timed query, so it never
      contributes a stored sample.
    * Classification and recall metrics are computed **per repetition** from
      that repetition's own neighbour sets, then summarised across
      repetitions. They are not computed once and copied.
    """
    started = time.perf_counter()

    dataset = fetch_one("SELECT * FROM datasets WHERE id = %s", (config.dataset_id,))
    if dataset is None:
        raise ValueError(f"Dataset {config.dataset_id} does not exist")

    state = PreprocessState.from_dict(dataset["preprocessing"])
    class_labels = state.class_labels

    selected_ids = vs.select_test_ids(
        config.dataset_id,
        config.max_queries,
        sampling=config.query_sampling,
        seed=config.query_sample_seed,
    )
    rows = vs.load_test_vectors_by_ids(config.dataset_id, selected_ids)
    if not rows:
        raise ValueError("Dataset has no test vectors to query")

    query_ids = [r[0] for r in rows]
    query_vectors = [r[2] for r in rows]
    y_true = [r[1] for r in rows]

    train_size = vs.count_vectors(config.dataset_id, "train")
    total_test_pool = vs.count_vectors(config.dataset_id, "test")
    pool_distribution = _class_distribution(vs.label_counts(config.dataset_id, "test"))
    evaluated_distribution = _class_distribution(y_true)

    # --- Index preparation ------------------------------------------------
    preparation_started = time.perf_counter()
    index_build_ms: float | None = None
    index_created_ms: float | None = None
    index_reused: bool | None = None
    index_reused_build_ms: float | None = None
    index_size: int | None = None
    index_m: int | None = None
    index_ef_construction: int | None = None

    if config.method == "hnsw":
        info = vs.ensure_hnsw_index(
            config.dataset_id,
            config.distance_metric,
            m=config.hnsw_m,
            ef_construction=config.hnsw_ef_construction,
            force=config.force_rebuild_index,
        )
        # The live index's parameters are authoritative, not the Python
        # defaults that were requested.
        index_m = info.m
        index_ef_construction = info.ef_construction
        index_size = info.size_bytes
        index_reused = not info.created
        if info.created:
            index_build_ms = info.build_ms
            index_created_ms = info.build_ms
        else:
            # A reused index was not built during this run. Its historical
            # build time is retained separately so it is never reported as a
            # cost paid by this ef_search configuration.
            index_reused_build_ms = info.build_ms
        vs.analyze(config.dataset_id)
    else:
        # The exact baseline uses the pgvector distance *function*, which the
        # planner can never satisfy with an index. Any existing HNSW index is
        # therefore left in place - it simply cannot be used - which keeps a
        # following HNSW run from paying the rebuild cost again.
        vs.analyze(config.dataset_id)
        existing = vs.get_hnsw_index(config.dataset_id)
        if existing is not None:
            index_m = int(existing["m"])
            index_ef_construction = int(existing["ef_construction"])
    index_preparation_s = time.perf_counter() - preparation_started

    # --- Ground truth (always exhaustive, tie-tolerant) --------------------
    ground_truth_started = time.perf_counter()
    tolerance = TieTolerance(
        relative=config.tie_tolerance_relative,
        absolute=config.tie_tolerance_absolute,
    )
    ground_truth = vs.ground_truth_candidates(
        query_vectors,
        config.k,
        config.distance_metric,
        config.dataset_id,
        tolerance,
    )
    gt_ids = [g.ids for g in ground_truth]
    gt_candidates = [g.candidate_ids for g in ground_truth]
    tied_queries = sum(1 for g in ground_truth if g.is_tied)
    max_tied_candidates = max((len(g.candidate_ids) for g in ground_truth), default=0)
    ground_truth_computation_s = time.perf_counter() - ground_truth_started

    # --- Retrieval under test ---------------------------------------------
    batch = vs.retrieve_batch(
        query_vectors,
        k=config.k,
        metric=config.distance_metric,
        dataset_id=config.dataset_id,
        method=config.method,
        ef_search=config.ef_search,
        repetitions=config.repetitions,
    )

    # --- Metrics per repetition -------------------------------------------
    metrics_started = time.perf_counter()
    per_repetition: list[dict[str, Any]] = []
    last_metrics: ClassificationMetrics | None = None

    for index, rep_results in enumerate(batch.repetition_results):
        retrieved = [[n.row_id for n in r.neighbours] for r in rep_results]
        y_pred = [
            majority_vote(
                [n.label for n in r.neighbours],
                [n.distance for n in r.neighbours],
                weighted=config.weighted_voting,
            )
            for r in rep_results
        ]
        metrics = evaluate_predictions(y_true, y_pred, class_labels)
        last_metrics = metrics
        rep_lat = latency_stats(batch.repetition_latencies[index])

        per_repetition.append(
            {
                "repetition": index + 1,
                "unique_queries": len(retrieved),
                "samples": len(batch.repetition_latencies[index]),
                "latency_avg_ms": rep_lat["avg"],
                "latency_p50_ms": rep_lat["p50"],
                "latency_p95_ms": rep_lat["p95"],
                "latency_p99_ms": rep_lat["p99"],
                "recall_at_k": recall_at_k(retrieved, gt_ids),
                "tie_aware_recall_at_k": tie_aware_recall_at_k(
                    retrieved, gt_candidates
                ),
                "accuracy": metrics.accuracy,
                "precision_macro": metrics.precision_macro,
                "recall_macro": metrics.recall_macro,
                "macro_f1": metrics.macro_f1,
            }
        )

    if last_metrics is None:  # pragma: no cover - guarded by repetitions >= 1
        raise RuntimeError("No repetition was executed")

    metrics_computation_s = time.perf_counter() - metrics_started

    # Latency percentiles come from every measured observation, not from a
    # per-repetition average.  The wall-clock time around the retrieval loop
    # is *not* used here: the sum of individually timed queries avoids
    # counting connection setup and result-conversion work twice.
    lat = latency_stats(batch.latencies)
    retrieval_total_s = sum(batch.latencies) / 1000.0
    retrieval_mean_per_query_ms = (
        sum(batch.latencies) / len(batch.latencies) if batch.latencies else 0.0
    )

    summary_keys = (
        "accuracy",
        "precision_macro",
        "recall_macro",
        "macro_f1",
        "recall_at_k",
        "tie_aware_recall_at_k",
    )
    summaries = {
        key: repetition_summary([rep[key] for rep in per_repetition])
        for key in summary_keys
    }

    used_index = batch.used_index_scan
    used_seq_scan = batch.used_sequential_scan
    applied_ef = batch.applied_ef_search

    fingerprint = dataset_fingerprint(config.dataset_id)
    index_build_s = (index_build_ms / 1000.0) if index_build_ms is not None else 0.0

    payload = {
        "dataset_id": config.dataset_id,
        "name": config.name,
        "retrieval_method": config.method,
        "k": config.k,
        "distance_metric": config.distance_metric.value,
        "hnsw_ef_search": config.ef_search,
        "hnsw_m": index_m,
        "hnsw_ef_construction": index_ef_construction,
        "train_size": train_size,
        "test_size": len(query_vectors),
        "unique_queries": len(query_vectors),
        "total_test_pool": total_test_pool,
        "index_build_ms": index_build_ms,
        "index_created_ms": index_created_ms,
        "index_reused": index_reused,
        "index_reused_build_ms": index_reused_build_ms,
        "index_size_bytes": index_size,
        "repetitions": config.repetitions,
        "recall_at_k": summaries["recall_at_k"]["mean"],
        "tie_aware_recall_at_k": summaries["tie_aware_recall_at_k"]["mean"],
        "tied_queries": tied_queries,
        "max_tied_candidates": max_tied_candidates,
        "accuracy": summaries["accuracy"]["mean"],
        "macro_f1": summaries["macro_f1"]["mean"],
        "precision_macro": summaries["precision_macro"]["mean"],
        "recall_macro": summaries["recall_macro"]["mean"],
        "accuracy_std": summaries["accuracy"]["sd"],
        "macro_f1_std": summaries["macro_f1"]["sd"],
        "recall_at_k_std": summaries["recall_at_k"]["sd"],
        "tie_aware_recall_at_k_std": summaries["tie_aware_recall_at_k"]["sd"],
        "dataset_fingerprint": fingerprint,
        "seed": config.seed,
        "tie_tolerance_rel": tolerance.relative,
        "tie_tolerance_abs": tolerance.absolute,
        "query_sampling": config.query_sampling,
        "query_sample_seed": config.query_sample_seed,
        "evaluated_query_ids": json.dumps(query_ids),
        "test_pool_class_distribution": json.dumps(pool_distribution),
        "evaluated_class_distribution": json.dumps(evaluated_distribution),
        "benchmark_round": config.benchmark_round,
        "benchmark_position": config.benchmark_position,
        "benchmark_seed": config.benchmark_seed,
        "benchmark_order": config.benchmark_order,
        "warm_cache": True,
        "latency_avg_ms": lat["avg"],
        "latency_p50_ms": lat["p50"],
        "latency_p95_ms": lat["p95"],
        "latency_p99_ms": lat["p99"],
        "latency_min_ms": lat["min"],
        "latency_max_ms": lat["max"],
        # Timing fields are filled in after persistence finishes (see below).
        "total_experiment_runtime_s": 0.0,
        "index_preparation_s": index_preparation_s,
        "index_build_s": index_build_s,
        "ground_truth_computation_s": ground_truth_computation_s,
        "retrieval_total_s": retrieval_total_s,
        "retrieval_mean_per_query_ms": retrieval_mean_per_query_ms,
        "metrics_computation_s": metrics_computation_s,
        "persistence_s": 0.0,
        "orchestration_overhead_s": 0.0,
        "total_runtime_s": 0.0,  # legacy alias, kept for older clients
        "used_index_scan": used_index,
        "environment": json.dumps(
            {
                **capture_environment(),
                "applied_ef_search": applied_ef,
                "used_sequential_scan": used_seq_scan,
                "measured_queries": batch.sample_count,
                "unique_queries": batch.unique_query_count,
                "repetitions": config.repetitions,
                "warmup_queries": batch.warmup_queries,
                "preprocessing_mode": state.mode,
                "preprocessing_dimension": state.dimension,
                "total_test_pool": total_test_pool,
                "dataset_fingerprint": fingerprint,
                "seed": config.seed,
                "repetition_query_order": "identical order across repetitions",
                "plan_verified": bool(batch.plan),
                "index_reused": index_reused,
                "query_sampling": config.query_sampling,
                "query_sample_seed": config.query_sample_seed,
                "query_count": len(query_ids),
                "benchmark_round": config.benchmark_round,
                "benchmark_position": config.benchmark_position,
                "benchmark_seed": config.benchmark_seed,
                "benchmark_cache": "warm buffer cache; no cold-cache procedure was executed",
            }
        ),
        "confusion_matrix": json.dumps(last_metrics.confusion_matrix),
        "per_class_metrics": json.dumps(last_metrics.per_class),
        "config": json.dumps(
            {
                **asdict(config),
                "distance_metric": config.distance_metric.value,
                "extra": config.extra,
            },
            default=str,
        ),
        "status": "completed",
    }

    persistence_started = time.perf_counter()
    experiment_id = _insert_experiment(payload)
    _insert_repetitions(experiment_id, per_repetition)
    _insert_latencies(experiment_id, batch)
    persistence_s = time.perf_counter() - persistence_started

    total_experiment_runtime_s = time.perf_counter() - started
    overlapping_phases = (
        index_preparation_s
        + ground_truth_computation_s
        + retrieval_total_s
        + metrics_computation_s
        + persistence_s
    )
    orchestration_overhead_s = max(0.0, total_experiment_runtime_s - overlapping_phases)

    timing_columns = {
        "total_experiment_runtime_s": total_experiment_runtime_s,
        "index_preparation_s": index_preparation_s,
        "index_build_s": index_build_s,
        "ground_truth_computation_s": ground_truth_computation_s,
        "retrieval_total_s": retrieval_total_s,
        "retrieval_mean_per_query_ms": retrieval_mean_per_query_ms,
        "metrics_computation_s": metrics_computation_s,
        "persistence_s": persistence_s,
        "orchestration_overhead_s": orchestration_overhead_s,
        "total_runtime_s": total_experiment_runtime_s,
    }
    assignments = ", ".join(f"{column} = %s" for column in timing_columns)
    execute(
        f"UPDATE experiments SET {assignments} WHERE id = %s",
        (*timing_columns.values(), experiment_id),
    )
    payload.update(timing_columns)

    logger.info(
        "Experiment %s done: method=%s k=%s ef=%s unique_queries=%s reps=%s "
        "samples=%s recall@%s=%.4f tie_aware=%.4f acc=%.4f p50=%.3fms idx=%s "
        "reused=%s",
        experiment_id,
        config.method,
        config.k,
        config.ef_search,
        batch.unique_query_count,
        config.repetitions,
        batch.sample_count,
        config.k,
        summaries["recall_at_k"]["mean"],
        summaries["tie_aware_recall_at_k"]["mean"],
        summaries["accuracy"]["mean"],
        lat["p50"],
        used_index,
        index_reused,
    )

    return {
        "id": experiment_id,
        **{
            key: value
            for key, value in payload.items()
            if key not in {"confusion_matrix", "per_class_metrics", "config", "environment"}
        },
        "confusion_matrix": last_metrics.confusion_matrix,
        "per_class_metrics": last_metrics.per_class,
        "plan": batch.plan,
        "per_repetition": per_repetition,
    }


def _insert_experiment(payload: dict[str, Any]) -> int:
    columns = ", ".join(payload.keys())
    placeholders = ", ".join(["%s"] * len(payload))
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"INSERT INTO experiments ({columns}) VALUES ({placeholders}) RETURNING id",
                tuple(payload.values()),
            )
            row = cur.fetchone()
            return int(row["id"])


def _insert_repetitions(experiment_id: int, per_repetition: list[dict[str, Any]]) -> None:
    """Persist one row per (experiment, repetition)."""
    if not per_repetition:
        return

    columns = (
        "experiment_id, repetition, unique_queries, samples, "
        "latency_avg_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms, "
        "recall_at_k, tie_aware_recall_at_k, accuracy, precision_macro, "
        "recall_macro, macro_f1"
    )
    with connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                f"INSERT INTO experiment_repetitions ({columns}) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
                [
                    (
                        experiment_id,
                        int(rep["repetition"]),
                        int(rep["unique_queries"]),
                        int(rep["samples"]),
                        rep["latency_avg_ms"],
                        rep["latency_p50_ms"],
                        rep["latency_p95_ms"],
                        rep["latency_p99_ms"],
                        rep["recall_at_k"],
                        rep["tie_aware_recall_at_k"],
                        rep["accuracy"],
                        rep["precision_macro"],
                        rep["recall_macro"],
                        rep["macro_f1"],
                    )
                    for rep in per_repetition
                ],
            )


def _insert_latencies(experiment_id: int, batch) -> None:
    """Persist every measured latency sample.

    One row is written per sample, so an experiment with ``repetitions``
    passes over ``n`` queries stores exactly ``repetitions * n`` rows. The
    ``repetition`` column records which pass produced the sample and
    ``query_index`` records which query. The warm-up query is not part of any
    pass and is therefore never stored.
    """
    rows: list[tuple] = []
    for repetition, (results, latencies) in enumerate(
        zip(batch.repetition_results, batch.repetition_latencies), start=1
    ):
        for query_index, (result, latency) in enumerate(zip(results, latencies)):
            rows.append(
                (
                    experiment_id,
                    repetition,
                    query_index,
                    float(latency),
                    json.dumps([n.row_id for n in result.neighbours]),
                )
            )

    if not rows:
        return

    with connection() as conn:
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO query_latencies "
                "(experiment_id, repetition, query_index, latency_ms, retrieved_ids) "
                "VALUES (%s, %s, %s, %s, %s::jsonb)",
                rows,
            )


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def compare_experiments(exact_id: int, approx_id: int) -> dict[str, Any]:
    """Compare two experiments row by row."""
    exact = fetch_one("SELECT * FROM experiments WHERE id = %s", (exact_id,))
    approx = fetch_one("SELECT * FROM experiments WHERE id = %s", (approx_id,))
    if exact is None or approx is None:
        raise ValueError("Both experiment ids must exist")

    def delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return float(b) - float(a)

    return {
        "exact": _serialize_experiment(exact),
        "approximate": _serialize_experiment(approx),
        "deltas": {
            "accuracy": delta(exact["accuracy"], approx["accuracy"]),
            "precision_macro": delta(exact["precision_macro"], approx["precision_macro"]),
            "recall_macro": delta(exact["recall_macro"], approx["recall_macro"]),
            "macro_f1": delta(exact["macro_f1"], approx["macro_f1"]),
            "recall_at_k": delta(exact["recall_at_k"], approx["recall_at_k"]),
            "tie_aware_recall_at_k": delta(
                exact["tie_aware_recall_at_k"], approx["tie_aware_recall_at_k"]
            ),
            "latency_avg_ms": delta(exact["latency_avg_ms"], approx["latency_avg_ms"]),
            "latency_p50_ms": delta(exact["latency_p50_ms"], approx["latency_p50_ms"]),
            "latency_p95_ms": delta(exact["latency_p95_ms"], approx["latency_p95_ms"]),
            "speedup": (
                exact["latency_avg_ms"] / approx["latency_avg_ms"]
                if approx["latency_avg_ms"]
                else None
            ),
        },
    }


def get_repetitions(experiment_id: int) -> list[dict[str, Any]]:
    """Per-repetition measurement rows for one experiment."""
    return fetch_all(
        "SELECT * FROM experiment_repetitions WHERE experiment_id = %s "
        "ORDER BY repetition",
        (experiment_id,),
    )


def _serialize_experiment(row: dict) -> dict[str, Any]:
    out = dict(row)
    for key in (
        "confusion_matrix",
        "per_class_metrics",
        "config",
        "environment",
        "evaluated_query_ids",
        "test_pool_class_distribution",
        "evaluated_class_distribution",
        "benchmark_order",
    ):
        raw = out.get(key)
        if isinstance(raw, str):
            try:
                out[key] = json.loads(raw)
            except json.JSONDecodeError:
                pass
    return out


def list_experiments(dataset_id: int | None = None, limit: int = 100) -> list[dict]:
    if dataset_id is None:
        rows = fetch_all(
            "SELECT * FROM experiments ORDER BY created_at DESC LIMIT %s", (limit,)
        )
    else:
        rows = fetch_all(
            "SELECT * FROM experiments WHERE dataset_id = %s "
            "ORDER BY created_at DESC LIMIT %s",
            (dataset_id, limit),
        )
    return [_serialize_experiment(r) for r in rows]


def get_experiment(experiment_id: int) -> dict | None:
    row = fetch_one("SELECT * FROM experiments WHERE id = %s", (experiment_id,))
    return _serialize_experiment(row) if row else None


def delete_experiment(experiment_id: int) -> None:
    execute("DELETE FROM experiments WHERE id = %s", (experiment_id,))
