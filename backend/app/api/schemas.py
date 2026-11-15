"""Pydantic request/response schemas for the API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------
class DatasetSummary(BaseModel):
    id: int
    name: str
    source: str
    n_rows: int
    n_features: int
    n_classes: int
    dimension: int
    class_labels: list[int]
    created_at: str


class DatasetIngestRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    target_column: str = "label"
    test_size: int | None = Field(None, ge=1, le=50_000)
    train_fraction: float = Field(0.7, gt=0, lt=1)
    max_train_rows: int | None = Field(None, ge=100)
    seed: int = 42


class CovertypeIngestRequest(BaseModel):
    name: str = "Forest CoverType"
    max_rows: int | None = Field(None, ge=100)
    test_size: int | None = Field(5_000, ge=1, le=50_000)
    max_train_rows: int | None = Field(None, ge=100)
    seed: int = 42


class IngestResponse(BaseModel):
    dataset_id: int
    name: str
    n_train: int
    n_test: int
    dimension: int
    n_classes: int
    classes: list[int]
    class_distribution: list[int]
    preprocessing: dict[str, Any]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
class ExperimentRequest(BaseModel):
    dataset_id: int
    k: int = Field(5, ge=1, le=100)
    distance_metric: Literal["l2", "cosine", "l1"] = "l2"
    method: Literal["exact", "hnsw"] = "exact"
    ef_search: int | None = Field(None, ge=1, le=1000)
    hnsw_m: int = Field(16, ge=2, le=128)
    hnsw_ef_construction: int = Field(64, ge=4, le=1000)
    max_queries: int | None = Field(2_000, ge=1, le=50_000)
    name: str | None = None
    weighted_voting: bool = False


class ExperimentSummary(BaseModel):
    id: int
    dataset_id: int
    name: str | None
    retrieval_method: str
    k: int
    distance_metric: str
    hnsw_ef_search: int | None
    train_size: int
    test_size: int
    recall_at_k: float | None
    accuracy: float | None
    macro_f1: float | None
    latency_avg_ms: float | None
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    index_build_ms: float | None
    status: str
    created_at: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    status: str
    database: str
    pgvector_version: str | None
    dataset_count: int
    experiment_count: int
