"""Phase 2 remediation regression tests.

These tests create only disposable ``pytest-phase2-*`` datasets and experiments
in a database whose name starts with ``test_``. They never reference dataset
116, dataset 846 or any historical result.

Run with::

    ALLOW_PHASE2_DB_TESTS=1 PHASE2_TEST_DATABASE_URL=<isolated URL> pytest \
        tests/test_phase2_remediation.py

The backend settings (``POSTGRES_DB``) must point at the same isolated test
database.
"""

from __future__ import annotations

import os
import uuid

import pytest

from app.core.database import close_pool, connection, execute, fetch_one, transaction
from app.db.init_db import init_schema
from app.services import vector_store as vs
from app.services.experiments import (
    ExperimentConfig,
    compare_experiments,
    run_experiment,
)
from app.services.metrics import DistanceMetric

pytestmark = pytest.mark.skipif(
    os.getenv("ALLOW_PHASE2_DB_TESTS") != "1",
    reason="set ALLOW_PHASE2_DB_TESTS=1 and PHASE2_TEST_DATABASE_URL for Phase 2 DB tests",
)


@pytest.fixture(scope="module", autouse=True)
def isolated_schema() -> None:
    assert os.getenv("PHASE2_TEST_DATABASE_URL"), "PHASE2_TEST_DATABASE_URL is required"
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database() AS db")
            db = str(cur.fetchone()["db"])
    assert db.startswith("test_"), f"refusing to run on non-test database {db!r}"
    init_schema()
    yield
    close_pool()


@pytest.fixture
def phase2_env():
    """Factory for disposable datasets; records everything for cleanup."""
    state: dict[str, list[int]] = {"datasets": [], "experiments": []}

    def make_dataset(*, n_train: int, n_test: int = 5, dim: int = 8, seed: int = 7) -> int:
        import numpy as np

        rng = np.random.default_rng(seed)
        train = [
            "[" + ",".join(f"{v:.6f}" for v in row) + "]"
            for row in rng.normal(size=(n_train, dim))
        ]
        test = [
            "[" + ",".join(f"{v:.6f}" for v in row) + "]"
            for row in rng.normal(size=(n_test, dim))
        ]
        labels = list(rng.integers(0, 2, size=n_train))
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO datasets (name, source, n_rows, n_features, n_classes, "
                    "dimension, class_labels, preprocessing, preprocessing_mode) "
                    "VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) "
                    "RETURNING id",
                    (
                        f"pytest-phase2-{uuid.uuid4().hex[:12]}",
                        n_train,
                        dim,
                        2,
                        dim,
                        "[0,1]",
                        '{"class_labels": [0,1]}',
                        "covertype",
                    ),
                )
                dataset_id = int(cur.fetchone()["id"])
        state["datasets"].append(dataset_id)
        vs.create_dataset_tables(dataset_id, dim)
        vs.insert_vectors(dataset_id, "train", train, labels, list(range(n_train)))
        vs.insert_vectors(dataset_id, "test", test, [0] * n_test, list(range(n_test)))
        return dataset_id

    def track_experiment(experiment_id: int) -> int:
        state["experiments"].append(int(experiment_id))
        return int(experiment_id)

    yield make_dataset, track_experiment

    for experiment_id in state["experiments"]:
        execute(
            "DELETE FROM query_latencies WHERE experiment_id = %s", (experiment_id,)
        )
        execute(
            "DELETE FROM experiment_repetitions WHERE experiment_id = %s",
            (experiment_id,),
        )
        execute("DELETE FROM experiments WHERE id = %s", (experiment_id,))
    for dataset_id in state["datasets"]:
        vs.drop_dataset_tables(dataset_id)
        execute("DELETE FROM hnsw_indexes WHERE dataset_id = %s", (dataset_id,))
        execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))


def _one_query(dataset_id: int) -> str:
    return vs.load_test_vectors(dataset_id, 1)[0][2]


class TestPhase2PlannerModes:
    def test_controlled_hnsw_selects_index_at_small_pool(self, phase2_env) -> None:
        make_dataset, _ = phase2_env
        dataset_id = make_dataset(n_train=1_000, n_test=5)
        vs.ensure_hnsw_index(dataset_id, DistanceMetric.L2)
        vector = _one_query(dataset_id)

        batch = vs.retrieve_batch(
            [vector],
            k=5,
            metric=DistanceMetric.L2,
            dataset_id=dataset_id,
            method="hnsw",
            ef_search=50,
            repetitions=1,
            planner_mode="controlled_hnsw",
        )

        assert batch.planner_mode == "controlled_hnsw"
        assert batch.planner_settings["enable_seqscan"] == "off"
        assert batch.used_index_scan is True
        assert batch.used_sequential_scan is False
        assert batch.index_scan_checks >= batch.sample_count
        assert all(r.used_index_scan for r in batch.results)
        assert "fv_%d_embedding_hnsw_idx" % dataset_id in batch.plan

    def test_natural_fallback_is_failed_and_not_a_valid_summary(
        self, phase2_env
    ) -> None:
        make_dataset, track = phase2_env
        # Match the diagnosed smoke geometry (54 dimensions); the crossover
        # depends on table width, so an 8-dim toy pool may legitimately keep
        # the index at ef=50.
        dataset_id = make_dataset(n_train=1_000, n_test=6, dim=54)
        exact = run_experiment(
            ExperimentConfig(
                dataset_id=dataset_id, method="exact", max_queries=5, name="phase2-exact"
            )
        )
        track(exact["id"])
        hnsw = run_experiment(
            ExperimentConfig(
                dataset_id=dataset_id,
                method="hnsw",
                ef_search=50,
                max_queries=5,
                name="phase2-natural-ef50",
            )
        )
        track(hnsw["id"])

        assert exact["status"] == "completed"
        assert hnsw["status"] == "failed"
        assert hnsw["error_message"]
        assert "Seq Scan" in hnsw["plan"]
        detail = fetch_one(
            "SELECT environment FROM experiments WHERE id = %s", (hnsw["id"],)
        )
        environment = detail["environment"]
        assert environment["valid_for_summary"] is False
        assert environment["used_sequential_scan"] is True
        assert environment["planner_mode"] == "natural"

        comparison = compare_experiments(exact["id"], hnsw["id"])
        assert comparison["valid"] is False
        assert all(value is None for value in comparison["deltas"].values())

    def test_ef_search_is_applied_and_restored_on_same_connection(
        self, phase2_env
    ) -> None:
        make_dataset, _ = phase2_env
        dataset_id = make_dataset(n_train=1_000, n_test=5)
        vs.ensure_hnsw_index(dataset_id, DistanceMetric.L2)
        vector = _one_query(dataset_id)

        result = vs.retrieve(
            vector,
            5,
            DistanceMetric.L2,
            dataset_id,
            method="hnsw",
            ef_search=400,
            planner_mode="natural",
        )
        assert result.applied_ef_search == 400

        with connection() as conn:
            with conn.cursor() as cur:
                previous = vs._set_ef_search(cur, 100)
                assert vs._read_ef_search(cur) == 100
                vs._restore_ef_search(cur, previous)
                assert vs._read_ef_search(cur) == previous

    def test_index_verification_checks_operator_class(self, phase2_env) -> None:
        make_dataset, _ = phase2_env
        dataset_id = make_dataset(n_train=1_000, n_test=5)
        vs.ensure_hnsw_index(dataset_id, DistanceMetric.L2)
        assert vs.verify_hnsw_index(dataset_id, DistanceMetric.L2).operator_class == (
            "vector_l2_ops"
        )
        with pytest.raises(ValueError, match="operator class"):
            vs.verify_hnsw_index(dataset_id, DistanceMetric.COSINE)

    def test_exact_search_stays_exhaustive_and_valid(self, phase2_env) -> None:
        make_dataset, track = phase2_env
        dataset_id = make_dataset(n_train=1_000, n_test=5)
        vs.ensure_hnsw_index(dataset_id, DistanceMetric.L2)
        result = run_experiment(
            ExperimentConfig(
                dataset_id=dataset_id,
                method="exact",
                max_queries=4,
                name="phase2-exact-with-index",
            )
        )
        track(result["id"])
        assert result["status"] == "completed"
        assert result["used_index_scan"] is False
        assert "Seq Scan" in result["plan"]


class TestPhase2ConfigurationAndPlanChecks:
    def test_controlled_mode_rejects_exact_method(self) -> None:
        with pytest.raises(ValueError, match="controlled_hnsw"):
            ExperimentConfig(dataset_id=1, method="exact", planner_mode="controlled_hnsw")

    def test_api_schema_rejects_controlled_without_hnsw(self) -> None:
        from pydantic import ValidationError

        from app.api.schemas import ExperimentRequest

        with pytest.raises(ValidationError):
            ExperimentRequest(dataset_id=1, method="exact", planner_mode="controlled_hnsw")

    def test_plan_check_uses_exact_index_name(self) -> None:
        good = (
            "Index Scan using fv_7_embedding_hnsw_idx on fv_7  (cost=1.0..2.0 rows=1)\n"
            "  Buffers: shared hit=1"
        )
        wrong = (
            "Index Scan using fv_7_pkey on fv_7  (cost=1.0..2.0 rows=1)\n"
            "  Buffers: shared hit=1"
        )
        assert vs.plan_uses_index_scan(good, "fv_7_embedding_hnsw_idx") is True
        assert vs.plan_uses_index_scan(wrong, "fv_7_embedding_hnsw_idx") is False

    def test_phase1_transaction_guarantee_intact(self) -> None:
        table = "pytest_phase2_tx_" + uuid.uuid4().hex[:10]
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute(f"CREATE TABLE {table} (id integer)")
        with pytest.raises(RuntimeError):
            with transaction() as conn:
                with conn.cursor() as cur:
                    cur.execute(f"INSERT INTO {table} VALUES (1)")
                raise RuntimeError("rollback")
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) AS c FROM " + table)
                assert int(cur.fetchone()["c"]) == 0
                assert conn.autocommit is True
        execute(f"DROP TABLE {table}")
