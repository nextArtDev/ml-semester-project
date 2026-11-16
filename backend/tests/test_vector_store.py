"""Integration tests against a live PostgreSQL + pgvector database.

These tests are skipped automatically when the database is unreachable, so
``pytest`` still passes on a machine without PostgreSQL.
"""

from __future__ import annotations

import os

import pytest

from app.core.database import close_pool, connection, fetch_all, fetch_one
from app.db.init_db import init_schema
from app.services import vector_store as vs
from app.services.metrics import DistanceMetric


def _db_available() -> bool:
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="PostgreSQL is not reachable"
)


@pytest.fixture(scope="module", autouse=True)
def schema() -> None:
    init_schema()
    yield
    close_pool()


@pytest.fixture
def dataset_id() -> int:
    """Create a small synthetic dataset and clean it up afterwards."""
    import numpy as np

    rng = np.random.default_rng(0)
    dim = 8
    # Large enough that the planner prefers an HNSW index scan over a
    # sequential scan, which is the behaviour the tests assert.
    n_train = 20_000

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasets (name, source, n_rows, n_features, n_classes, "
                "dimension, class_labels, preprocessing) "
                "VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb) RETURNING id",
                (
                    "pytest-synthetic",
                    n_train,
                    dim,
                    3,
                    dim,
                    "[0, 1, 2]",
                    '{"class_labels": [0, 1, 2], "label_to_index": {"0": 0, "1": 1, "2": 2}}',
                ),
            )
            did = int(cur.fetchone()["id"])

    vs.create_dataset_tables(did, dim)

    embeddings = [
        "[" + ",".join(f"{v:.6f}" for v in row) + "]"
        for row in rng.normal(size=(n_train, dim))
    ]
    labels = list(rng.integers(0, 3, size=n_train))
    vs.insert_vectors(did, "train", embeddings, labels, list(range(n_train)))

    test_embeddings = embeddings[:50]
    test_labels = labels[:50]
    vs.insert_vectors(did, "test", test_embeddings, test_labels, list(range(50)))

    yield did

    vs.drop_dataset_tables(did)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE id = %s", (did,))


class TestVectorStore:
    def test_tables_created(self, dataset_id: int) -> None:
        assert vs.count_vectors(dataset_id, "train") == 20_000
        assert vs.count_vectors(dataset_id, "test") == 50

    def test_exact_retrieval_returns_k_neighbours(self, dataset_id: int) -> None:
        rows = vs.load_test_vectors(dataset_id, 10)
        vector = rows[0][2]
        result = vs.retrieve(
            vector, 5, DistanceMetric.L2, dataset_id, method="exact"
        )
        assert len(result.neighbours) == 5
        assert result.used_index_scan is False
        distances = [n.distance for n in result.neighbours]
        assert distances == sorted(distances)

    def test_exact_result_is_self_match(self, dataset_id: int) -> None:
        """The nearest neighbour of a training row is the row itself."""
        rows = vs.load_test_vectors(dataset_id, 1)
        result = vs.retrieve(
            rows[0][2], 1, DistanceMetric.L2, dataset_id, method="exact"
        )
        assert result.neighbours[0].distance == pytest.approx(0.0, abs=1e-6)

    def test_hnsw_index_builds_and_is_used(self, dataset_id: int) -> None:
        build_ms = vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        assert build_ms > 0
        assert vs.hnsw_index_exists(dataset_id)

        rows = vs.load_test_vectors(dataset_id, 5)
        result = vs.retrieve(
            rows[0][2],
            5,
            DistanceMetric.L2,
            dataset_id,
            method="hnsw",
            ef_search=100,
        )
        assert result.used_index_scan is True
        assert len(result.neighbours) == 5

    def test_hnsw_index_is_reused(self, dataset_id: int) -> None:
        first = vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        second = vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        assert first == pytest.approx(second)

    def test_exact_ground_truth_matches_hnsw_at_high_ef(self, dataset_id: int) -> None:
        """A large ef_search should reproduce the exact neighbour set."""
        vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        vectors = [r[2] for r in vs.load_test_vectors(dataset_id, 20)]

        truth = vs.ground_truth_ids(vectors, 5, DistanceMetric.L2, dataset_id)

        approx = vs.retrieve_batch(
            vectors, 5, DistanceMetric.L2, dataset_id, method="hnsw", ef_search=500
        )
        for got, want in zip(approx, truth):
            got_ids = [n.row_id for n in got.neighbours]
            assert set(got_ids) == set(want)

    def test_higher_ef_search_yields_at_least_as_much_recall(
        self, dataset_id: int
    ) -> None:
        from app.services.classification import recall_at_k

        vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        vectors = [r[2] for r in vs.load_test_vectors(dataset_id, 30)]
        truth = vs.ground_truth_ids(vectors, 5, DistanceMetric.L2, dataset_id)

        def recall_for(ef: int) -> float:
            results = vs.retrieve_batch(
                vectors, 5, DistanceMetric.L2, dataset_id, method="hnsw", ef_search=ef
            )
            got = [[n.row_id for n in r.neighbours] for r in results]
            return recall_at_k(got, truth)

        assert recall_for(400) >= recall_for(10) - 1e-9


class TestSchema:
    def test_hnsw_registry_table_exists(self) -> None:
        row = fetch_one(
            "SELECT count(*) AS c FROM information_schema.tables "
            "WHERE table_name = 'hnsw_indexes'"
        )
        assert int(row["c"]) == 1

    def test_experiments_table_exists(self) -> None:
        rows = fetch_all("SELECT id FROM experiments LIMIT 1")
        assert isinstance(rows, list)
