"""Tests for the experiment timing decomposition.

The old ``api_overhead_s`` was total runtime minus query time, which silently
charged ground-truth computation, index preparation, metrics, persistence and
orchestration to "API overhead".  These tests check that the new fields are
non-overlapping, non-negative, and consistent with the stored latencies.
"""

from __future__ import annotations

import math

import pytest

from app.core.database import close_pool, connection
from app.db.init_db import init_schema
from app.services import vector_store as vs
from app.services.experiments import ExperimentConfig, run_experiment


def _db_available() -> bool:
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                return cur.fetchone() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="PostgreSQL with pgvector is not reachable"
)


@pytest.fixture(scope="module", autouse=True)
def schema() -> None:
    init_schema()
    yield
    close_pool()


@pytest.fixture
def timing_dataset() -> int:
    dim = 6
    n = 300
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasets (name, source, n_rows, n_features, n_classes, "
                "dimension, class_labels, preprocessing, preprocessing_mode) "
                "VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) "
                "RETURNING id",
                (
                    "pytest-timing",
                    n,
                    dim,
                    3,
                    dim,
                    "[0,1,2]",
                    '{"class_labels": [0,1,2], "label_to_index": {"0":0,"1":1,"2":2}}',
                    "covertype",
                ),
            )
            dataset_id = int(cur.fetchone()["id"])

    vs.create_dataset_tables(dataset_id, dim)
    vs.insert_vectors(
        dataset_id,
        "train",
        ["[" + ",".join([str((i + j) % 3) for j in range(dim)]) + "]" for i in range(n)],
        [i % 3 for i in range(n)],
        list(range(n)),
    )
    vs.insert_vectors(
        dataset_id,
        "test",
        ["[" + ",".join([str((i + j + 1) % 3) for j in range(dim)]) + "]" for i in range(60)],
        [i % 3 for i in range(60)],
        list(range(60)),
    )
    yield dataset_id
    vs.drop_dataset_tables(dataset_id)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))


PHASES = (
    "index_preparation_s",
    "ground_truth_computation_s",
    "retrieval_total_s",
    "metrics_computation_s",
    "persistence_s",
    "orchestration_overhead_s",
)


class TestTimingAccounting:
    def test_phases_are_nonnegative_and_do_not_exceed_total(
        self, timing_dataset: int
    ) -> None:
        result = run_experiment(
            ExperimentConfig(
                dataset_id=timing_dataset,
                method="hnsw",
                ef_search=20,
                k=3,
                max_queries=40,
                repetitions=2,
            )
        )
        total = result["total_experiment_runtime_s"]
        assert total > 0
        for field in PHASES:
            assert result[field] >= 0
        assert sum(result[field] for field in PHASES) <= total + 1e-6

        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT " + ", ".join(PHASES) + ", total_experiment_runtime_s, "
                    "api_overhead_s FROM experiments WHERE id = %s",
                    (result["id"],),
                )
                stored = cur.fetchone()
        for field in PHASES:
            assert math.isclose(float(stored[field]), float(result[field]), rel_tol=1e-9)
        # The misleading legacy value is not written for new runs.
        assert stored["api_overhead_s"] is None

    def test_retrieval_totals_match_stored_samples(self, timing_dataset: int) -> None:
        result = run_experiment(
            ExperimentConfig(
                dataset_id=timing_dataset,
                method="exact",
                k=3,
                max_queries=30,
                repetitions=3,
            )
        )
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) AS c, sum(latency_ms) AS total, avg(latency_ms) AS avg "
                    "FROM query_latencies WHERE experiment_id = %s",
                    (result["id"],),
                )
                row = cur.fetchone()
        assert int(row["c"]) == 90
        assert math.isclose(
            float(result["retrieval_total_s"]), float(row["total"]) / 1000.0, rel_tol=1e-9
        )
        assert math.isclose(
            float(result["retrieval_mean_per_query_ms"]),
            float(row["avg"]),
            rel_tol=1e-9,
        )

    def test_fresh_build_and_reuse_are_separated(self, timing_dataset: int) -> None:
        fresh = run_experiment(
            ExperimentConfig(
                dataset_id=timing_dataset,
                method="hnsw",
                ef_search=10,
                k=3,
                max_queries=20,
                repetitions=1,
            )
        )
        reused = run_experiment(
            ExperimentConfig(
                dataset_id=timing_dataset,
                method="hnsw",
                ef_search=50,
                k=3,
                max_queries=20,
                repetitions=1,
            )
        )
        assert fresh["index_reused"] is False
        assert fresh["index_build_s"] > 0
        assert 0 < fresh["index_build_s"] <= fresh["index_preparation_s"] + 1e-6

        assert reused["index_reused"] is True
        assert reused["index_build_s"] == 0.0
        assert reused["index_reused_build_ms"] == fresh["index_build_ms"]

    def test_warm_cache_flag_is_recorded(self, timing_dataset: int) -> None:
        result = run_experiment(
            ExperimentConfig(
                dataset_id=timing_dataset,
                method="exact",
                k=3,
                max_queries=10,
                repetitions=1,
            )
        )
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT warm_cache, environment->>'benchmark_cache' AS cache "
                    "FROM experiments WHERE id = %s",
                    (result["id"],),
                )
                row = cur.fetchone()
        assert row["warm_cache"] is True
        assert "no cold-cache procedure" in row["cache"]
