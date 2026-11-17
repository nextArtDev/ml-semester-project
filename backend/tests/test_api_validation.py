"""Tests for API input validation.

These use FastAPI's TestClient, which exercises the real request-validation
layer without needing a running server. Requests that reach a handler are
not made in these tests, so no database is required.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def cleanup_created_datasets():
    """Remove any dataset this module creates.

    Some validation tests deliberately reach the handler (a 400 rather than a
    422 proves that validation passed), which means they really ingest a small
    dataset. Those rows must not survive the test run.
    """
    from app.core.database import execute, fetch_all, fetch_one
    from app.services import vector_store as vs

    before = fetch_one("SELECT COALESCE(max(id), 0) AS m FROM datasets")
    highest = int(before["m"])

    yield

    for row in fetch_all("SELECT id FROM datasets WHERE id > %s", (highest,)):
        dataset_id = int(row["id"])
        try:
            vs.drop_dataset_tables(dataset_id)
        except Exception:  # tables may never have been created
            pass
        execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))


class TestExperimentRequestValidation:
    def test_k_must_be_positive(self) -> None:
        response = client.post(
            "/api/experiments/run",
            json={"dataset_id": 1, "k": 0, "method": "exact"},
        )
        assert response.status_code == 422

    def test_k_above_limit_rejected(self) -> None:
        response = client.post(
            "/api/experiments/run",
            json={"dataset_id": 1, "k": 101, "method": "exact"},
        )
        assert response.status_code == 422

    def test_unknown_method_rejected(self) -> None:
        response = client.post(
            "/api/experiments/run",
            json={"dataset_id": 1, "method": "annoy"},
        )
        assert response.status_code == 422

    def test_unknown_metric_rejected(self) -> None:
        response = client.post(
            "/api/experiments/run",
            json={"dataset_id": 1, "distance_metric": "jaccard"},
        )
        assert response.status_code == 422

    def test_hnsw_without_ef_search_rejected(self) -> None:
        response = client.post(
            "/api/experiments/run",
            json={"dataset_id": 1, "method": "hnsw"},
        )
        assert response.status_code == 422
        assert "ef_search" in response.text

    def test_ef_search_below_range_rejected(self) -> None:
        response = client.post(
            "/api/experiments/run",
            json={"dataset_id": 1, "method": "hnsw", "ef_search": 0},
        )
        assert response.status_code == 422

    def test_repetitions_must_be_positive(self) -> None:
        response = client.post(
            "/api/experiments/run",
            json={"dataset_id": 1, "method": "exact", "repetitions": 0},
        )
        assert response.status_code == 422

    def test_repetitions_above_limit_rejected(self) -> None:
        response = client.post(
            "/api/experiments/run",
            json={"dataset_id": 1, "method": "exact", "repetitions": 101},
        )
        assert response.status_code == 422

    def test_valid_exact_request_passes_validation(self) -> None:
        # The handler will fail because dataset 999999 does not exist, but a
        # 400 (not 422) proves that validation passed.
        response = client.post(
            "/api/experiments/run",
            json={"dataset_id": 999999, "method": "exact", "repetitions": 3},
        )
        assert response.status_code == 400


class TestCovertypeRequestValidation:
    def test_invalid_preprocessing_mode_rejected(self) -> None:
        response = client.post(
            "/api/datasets/covertype",
            json={"preprocessing_mode": "unknown-mode"},
        )
        assert response.status_code == 422

    def test_valid_preprocessing_modes_accepted(self) -> None:
        for mode in ("covertype", "standardize_all"):
            response = client.post(
                "/api/datasets/covertype",
                json={"preprocessing_mode": mode, "max_rows": 100},
            )
            # A 422 would mean the mode string failed validation. A 400 or
            # 502 means validation passed and the handler ran (a 100-row
            # sample may legitimately contain a single class).
            assert response.status_code != 422

    def test_test_size_must_be_positive(self) -> None:
        response = client.post(
            "/api/datasets/covertype",
            json={"test_size": 0},
        )
        assert response.status_code == 422

    def test_seed_is_accepted(self) -> None:
        response = client.post(
            "/api/datasets/covertype",
            json={"seed": 123, "max_rows": 100},
        )
        assert response.status_code != 422

    def test_max_rows_below_minimum_rejected(self) -> None:
        response = client.post(
            "/api/datasets/covertype",
            json={"max_rows": 10},
        )
        assert response.status_code == 422


class TestExperimentConfigValidation:
    def test_invalid_method_rejected(self) -> None:
        from app.services.experiments import ExperimentConfig

        with pytest.raises(ValueError, match="method"):
            ExperimentConfig(dataset_id=1, method="annoy")

    def test_hnsw_requires_ef_search(self) -> None:
        from app.services.experiments import ExperimentConfig

        with pytest.raises(ValueError, match="ef_search"):
            ExperimentConfig(dataset_id=1, method="hnsw")

    def test_repetitions_rejected(self) -> None:
        from app.services.experiments import ExperimentConfig

        with pytest.raises(ValueError, match="repetitions"):
            ExperimentConfig(dataset_id=1, method="exact", repetitions=0)
