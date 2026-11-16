"""Experiment orchestration.

An *experiment* is one retrieval configuration evaluated against one
dataset. The service:

1. builds (or reuses) the HNSW index,
2. runs the exact baseline to obtain ground-truth neighbours,
3. runs the configured retrieval method,
4. classifies both, compares accuracy / macro-F1 / recall@K / latency,
5. persists everything to the ``experiments`` and ``query_latencies`` tables.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from app.core.database import connection, execute, fetch_all, fetch_one
from app.services import vector_store as vs
from app.services.classification import (
    ClassificationMetrics,
    evaluate_predictions,
    latency_stats,
    majority_vote,
    recall_at_k,
)
from app.services.metrics import DistanceMetric
from app.services.preprocessing import PreprocessState

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""

    dataset_id: int
    k: int = 5
    distance_metric: DistanceMetric = DistanceMetric.L2
    method: str = "exact"  # 'exact' | 'hnsw'
    ef_search: int | None = None
    hnsw_m: int = 16
    hnsw_ef_construction: int = 64
    hnsw_lists: int = 100
    max_queries: int | None = 2_000
    name: str | None = None
    weighted_voting: bool = False
    use_operator: bool = True
    extra: dict[str, Any] = field(default_factory=dict)


def _load_test_vectors(dataset_id: int, limit: int | None) -> list[tuple[int, int, str]]:
    return vs.load_test_vectors(dataset_id, limit)


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    """Execute an experiment and persist its results."""
    started = time.perf_counter()

    dataset = fetch_one("SELECT * FROM datasets WHERE id = %s", (config.dataset_id,))
    if dataset is None:
        raise ValueError(f"Dataset {config.dataset_id} does not exist")

    state = PreprocessState.from_dict(dataset["preprocessing"])
    class_labels = state.class_labels

    rows = _load_test_vectors(config.dataset_id, config.max_queries)
    if not rows:
        raise ValueError("Dataset has no test vectors to query")

    query_vectors = [r[2] for r in rows]
    y_true = [r[1] for r in rows]

    train_size = vs.count_vectors(config.dataset_id, "train")

    # --- Index preparation ------------------------------------------------
    index_build_ms: float | None = None
    if config.method == "hnsw":
        index_build_ms = vs.create_hnsw_index(
            config.dataset_id,
            config.distance_metric,
            m=config.hnsw_m,
            ef_construction=config.hnsw_ef_construction,
        )
        vs.analyze(config.dataset_id)
    else:
        # The exact baseline uses the pgvector distance *function*, which the
        # planner can never satisfy with an index. Any existing HNSW index is
        # therefore left in place - it simply cannot be used - which keeps a
        # following HNSW run from paying the rebuild cost again.
        vs.analyze(config.dataset_id)

    # --- Ground truth (always exact) -------------------------------------
    gt_ids = vs.ground_truth_ids(
        query_vectors, config.k, config.distance_metric, config.dataset_id
    )

    # --- Retrieval under test --------------------------------------------
    results = vs.retrieve_batch(
        query_vectors,
        k=config.k,
        metric=config.distance_metric,
        dataset_id=config.dataset_id,
        method=config.method,
        ef_search=config.ef_search,
    )

    retrieved_ids = [[n.row_id for n in r.neighbours] for r in results]
    latencies = [r.latency_ms for r in results]

    y_pred = [
        majority_vote(
            [n.label for n in r.neighbours],
            [n.distance for n in r.neighbours],
            weighted=config.weighted_voting,
        )
        for r in results
    ]

    metrics: ClassificationMetrics = evaluate_predictions(y_true, y_pred, class_labels)
    recall = recall_at_k(retrieved_ids, gt_ids)
    lat = latency_stats(latencies)

    used_index = any(r.used_index_scan for r in results)
    total_runtime = time.perf_counter() - started

    payload = {
        "dataset_id": config.dataset_id,
        "name": config.name,
        "retrieval_method": config.method,
        "k": config.k,
        "distance_metric": config.distance_metric.value,
        "hnsw_ef_search": config.ef_search,
        "hnsw_m": config.hnsw_m if config.method == "hnsw" else None,
        "hnsw_lists": config.hnsw_lists if config.method == "hnsw" else None,
        "train_size": train_size,
        "test_size": len(query_vectors),
        "index_build_ms": index_build_ms,
        "recall_at_k": recall,
        "accuracy": metrics.accuracy,
        "macro_f1": metrics.macro_f1,
        "precision_macro": metrics.precision_macro,
        "recall_macro": metrics.recall_macro,
        "latency_avg_ms": lat["avg"],
        "latency_p50_ms": lat["p50"],
        "latency_p95_ms": lat["p95"],
        "latency_p99_ms": lat["p99"],
        "total_runtime_s": total_runtime,
        "used_index_scan": used_index,
        "confusion_matrix": json.dumps(metrics.confusion_matrix),
        "per_class_metrics": json.dumps(metrics.per_class),
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

    experiment_id = _insert_experiment(payload)
    _insert_latencies(experiment_id, results)

    logger.info(
        "Experiment %s done: method=%s k=%s acc=%.4f recall@%s=%.4f p50=%.3fms",
        experiment_id,
        config.method,
        config.k,
        metrics.accuracy,
        config.k,
        recall,
        lat["p50"],
    )

    return {
        "id": experiment_id,
        **{key: value for key, value in payload.items() if key not in {"confusion_matrix", "per_class_metrics", "config"}},
        "confusion_matrix": metrics.confusion_matrix,
        "per_class_metrics": metrics.per_class,
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


def _insert_latencies(experiment_id: int, results) -> None:
    with connection() as conn:
        with conn.cursor() as cur:
            for i, r in enumerate(results):
                cur.execute(
                    "INSERT INTO query_latencies "
                    "(experiment_id, query_index, latency_ms, retrieved_ids) "
                    "VALUES (%s, %s, %s, %s::jsonb)",
                    (
                        experiment_id,
                        i,
                        r.latency_ms,
                        json.dumps([n.row_id for n in r.neighbours]),
                    ),
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
            "macro_f1": delta(exact["macro_f1"], approx["macro_f1"]),
            "recall_at_k": delta(exact["recall_at_k"], approx["recall_at_k"]),
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


def _serialize_experiment(row: dict) -> dict[str, Any]:
    out = dict(row)
    for key in ("confusion_matrix", "per_class_metrics", "config"):
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
