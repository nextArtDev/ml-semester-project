"""Experiment execution and comparison endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.schemas import ExperimentRequest, ExperimentSummary
from app.core.database import fetch_all
from app.services.experiments import (
    ExperimentConfig,
    compare_experiments,
    delete_experiment,
    get_experiment,
    list_experiments,
    run_experiment,
)
from app.services.metrics import DistanceMetric

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.get("")
def list_all(dataset_id: int | None = None, limit: int = 100) -> list[dict]:
    return list_experiments(dataset_id=dataset_id, limit=limit)


@router.post("/run")
def run(request: ExperimentRequest) -> dict:
    """Run one experiment (exact baseline or HNSW approximate search)."""
    config = ExperimentConfig(
        dataset_id=request.dataset_id,
        k=request.k,
        distance_metric=DistanceMetric(request.distance_metric),
        method=request.method,
        ef_search=request.ef_search,
        hnsw_m=request.hnsw_m,
        hnsw_ef_construction=request.hnsw_ef_construction,
        max_queries=request.max_queries,
        repetitions=request.repetitions,
        name=request.name,
        weighted_voting=request.weighted_voting,
    )
    try:
        return run_experiment(config)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        logger.exception("Experiment failed")
        raise HTTPException(500, f"Experiment failed: {exc}") from exc


@router.get("/compare")
def compare(exact_id: int, approximate_id: int) -> dict:
    try:
        return compare_experiments(exact_id, approximate_id)
    except ValueError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/{experiment_id}")
def get_one(experiment_id: int) -> dict:
    row = get_experiment(experiment_id)
    if row is None:
        raise HTTPException(404, "Experiment not found")
    return row


@router.delete("/{experiment_id}")
def remove(experiment_id: int) -> dict:
    delete_experiment(experiment_id)
    return {"deleted": experiment_id}


@router.get("/{experiment_id}/latencies")
def latency_samples(experiment_id: int, limit: int = 5_000) -> dict:
    rows = fetch_all(
        "SELECT query_index, latency_ms FROM query_latencies "
        "WHERE experiment_id = %s ORDER BY query_index LIMIT %s",
        (experiment_id, limit),
    )
    values = [float(r["latency_ms"]) for r in rows]
    if not values:
        return {"experiment_id": experiment_id, "samples": [], "count": 0}

    ordered = sorted(values)
    def pct(p: float) -> float:
        idx = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
        return ordered[idx]

    return {
        "experiment_id": experiment_id,
        "count": len(values),
        "samples": values,
        "p50": pct(50),
        "p90": pct(90),
        "p95": pct(95),
        "p99": pct(99),
    }


@router.get("/{experiment_id}/explain")
def explain(experiment_id: int) -> dict:
    """Return the retrieval query used by an experiment, for inspection."""
    row = get_experiment(experiment_id)
    if row is None:
        raise HTTPException(404, "Experiment not found")
    return {
        "experiment_id": experiment_id,
        "retrieval_method": row["retrieval_method"],
        "k": row["k"],
        "distance_metric": row["distance_metric"],
        "hnsw_ef_search": row["hnsw_ef_search"],
        "used_index_scan": row["used_index_scan"],
        "config": row["config"],
    }
