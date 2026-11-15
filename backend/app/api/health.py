"""Health and system-info endpoints."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.database import fetch_one
from app.api.schemas import HealthResponse

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    pgvector_version = None
    database = "disconnected"
    dataset_count = 0
    experiment_count = 0

    try:
        row = fetch_one(
            """
            SELECT extversion AS version,
                   (SELECT count(*) FROM datasets) AS datasets,
                   (SELECT count(*) FROM experiments) AS experiments
            FROM pg_extension WHERE extname = 'vector'
            """
        )
        if row is not None:
            pgvector_version = row["version"]
            database = "connected"
            dataset_count = int(row["datasets"])
            experiment_count = int(row["experiments"])
    except Exception:
        database = "error"

    return HealthResponse(
        status="ok" if database == "connected" else "degraded",
        database=database,
        pgvector_version=pgvector_version,
        dataset_count=dataset_count,
        experiment_count=experiment_count,
    )
