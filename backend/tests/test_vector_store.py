"""Integration tests for vector storage and retrieval.

These tests use a real PostgreSQL database with pgvector enabled. They are
skipped automatically when the database is unreachable, so the unit tests
still run on a machine without PostgreSQL. Every executed test below is a
real database integration test — nothing is mocked.
"""

from __future__ import annotations

import pytest

from app.core.database import close_pool, connection, fetch_one
from app.db.init_db import init_schema
from app.services import vector_store as vs
from app.services.classification import recall_at_k
from app.services.metrics import DistanceMetric


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
                "dimension, class_labels, preprocessing, preprocessing_mode) "
                "VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) RETURNING id",
                (
                    "pytest-synthetic",
                    n_train,
                    dim,
                    3,
                    dim,
                    "[0, 1, 2]",
                    '{"class_labels": [0, 1, 2], "label_to_index": {"0": 0, "1": 1, "2": 2}}',
                    "covertype",
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
    vs.insert_vectors(did, "test", embeddings[:50], labels[:50], list(range(50)))

    yield did

    vs.drop_dataset_tables(did)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE id = %s", (did,))


class TestTableLifecycle:
    def test_tables_created(self, dataset_id: int) -> None:
        assert vs.count_vectors(dataset_id, "train") == 20_000
        assert vs.count_vectors(dataset_id, "test") == 50

    def test_deletion_cleans_up(self) -> None:
        """Deleting a dataset's tables leaves no residue."""
        import numpy as np

        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO datasets (name, source, dimension) "
                    "VALUES ('pytest-delete', 'test', 4) RETURNING id"
                )
                did = int(cur.fetchone()["id"])
        vs.create_dataset_tables(did, 4)
        assert vs.hnsw_index_exists(did) is False
        vs.drop_dataset_tables(did)
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT count(*) AS c FROM information_schema.tables "
                    "WHERE table_name = %s",
                    (vs.train_table(did),),
                )
                assert int(cur.fetchone()["c"]) == 0
                cur.execute("DELETE FROM datasets WHERE id = %s", (did,))

    def test_dimension_is_fixed(self, dataset_id: int) -> None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT format_type(atttypid, atttypmod) AS t "
                    "FROM pg_attribute "
                    "WHERE attrelid = %s::regclass AND attname = 'embedding'",
                    (vs.train_table(dataset_id),),
                )
                assert cur.fetchone()["t"] == "vector(8)"


class TestExactRetrieval:
    def test_returns_k_neighbours_in_distance_order(self, dataset_id: int) -> None:
        row = vs.load_test_vectors(dataset_id, 10)[0]
        result = vs.retrieve(
            row[2], 5, DistanceMetric.L2, dataset_id, method="exact"
        )
        assert len(result.neighbours) == 5
        distances = [n.distance for n in result.neighbours]
        assert distances == sorted(distances)

    def test_self_is_nearest(self, dataset_id: int) -> None:
        row = vs.load_test_vectors(dataset_id, 1)[0]
        result = vs.retrieve(row[2], 1, DistanceMetric.L2, dataset_id, method="exact")
        assert result.neighbours[0].distance == pytest.approx(0.0, abs=1e-6)

    def test_exact_plan_is_a_sequential_scan(self, dataset_id: int) -> None:
        row = vs.load_test_vectors(dataset_id, 1)[0]
        result = vs.retrieve(row[2], 5, DistanceMetric.L2, dataset_id, method="exact")
        assert result.used_sequential_scan is True
        assert result.used_index_scan is False
        assert "buffers:" in result.plan.lower()

    def test_exact_falls_back_when_index_exists(self, dataset_id: int) -> None:
        """The exact query cannot use the HNSW index, even when one exists."""
        vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        row = vs.load_test_vectors(dataset_id, 1)[0]
        result = vs.retrieve(row[2], 5, DistanceMetric.L2, dataset_id, method="exact")
        assert result.used_index_scan is False
        assert result.used_sequential_scan is True


class TestHNSWRetrieval:
    def test_index_builds_and_is_used(self, dataset_id: int) -> None:
        build_ms = vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        assert build_ms > 0
        assert vs.hnsw_index_exists(dataset_id)
        row = vs.load_test_vectors(dataset_id, 5)[0]
        result = vs.retrieve(
            row[2], 5, DistanceMetric.L2, dataset_id, method="hnsw", ef_search=100
        )
        assert result.used_index_scan is True
        assert "buffers:" in result.plan.lower()
        assert len(result.neighbours) == 5

    def test_index_size_is_recorded(self, dataset_id: int) -> None:
        vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        size = vs.index_size_bytes(dataset_id)
        assert size is not None and size > 0
        record = vs.get_hnsw_index(dataset_id)
        assert record is not None
        assert int(record["size_bytes"]) == size

    def test_index_is_reused(self, dataset_id: int) -> None:
        first = vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        second = vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        assert first == pytest.approx(second)

    def test_index_is_rebuilt_when_parameters_change(self, dataset_id: int) -> None:
        vs.create_hnsw_index(dataset_id, DistanceMetric.L2, m=16, ef_construction=64)
        record_before = vs.get_hnsw_index(dataset_id)
        vs.create_hnsw_index(dataset_id, DistanceMetric.L2, m=24, ef_construction=96)
        record_after = vs.get_hnsw_index(dataset_id)
        assert int(record_before["m"]) == 16
        assert int(record_after["m"]) == 24
        assert int(record_after["ef_construction"]) == 96

    def test_ef_search_is_applied_and_reported(self, dataset_id: int) -> None:
        vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        row = vs.load_test_vectors(dataset_id, 1)[0]
        for ef in (10, 100, 400):
            result = vs.retrieve(
                row[2],
                5,
                DistanceMetric.L2,
                dataset_id,
                method="hnsw",
                ef_search=ef,
            )
            assert result.applied_ef_search == ef

    def test_ef_search_does_not_leak_between_queries(self, dataset_id: int) -> None:
        vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        row = vs.load_test_vectors(dataset_id, 1)[0]
        vs.retrieve(row[2], 5, DistanceMetric.L2, dataset_id, method="hnsw", ef_search=400)
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SHOW hnsw.ef_search")
                assert int(cur.fetchone()["hnsw.ef_search"]) == 40  # pgvector default

    def test_high_ef_reproduces_exact_neighbours(self, dataset_id: int) -> None:
        vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        vectors = [r[2] for r in vs.load_test_vectors(dataset_id, 20)]
        truth = vs.ground_truth_ids(vectors, 5, DistanceMetric.L2, dataset_id)
        batch = vs.retrieve_batch(
            vectors, 5, DistanceMetric.L2, dataset_id, method="hnsw", ef_search=800
        )
        for got, want in zip(batch.results, truth):
            assert set(n.row_id for n in got.neighbours) == set(want)

    def test_recall_increases_with_ef_search(self, dataset_id: int) -> None:
        vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        vectors = [r[2] for r in vs.load_test_vectors(dataset_id, 30)]
        truth = vs.ground_truth_ids(vectors, 5, DistanceMetric.L2, dataset_id)

        def recall_for(ef: int) -> float:
            batch = vs.retrieve_batch(
                vectors, 5, DistanceMetric.L2, dataset_id, method="hnsw", ef_search=ef
            )
            got = [[n.row_id for n in r.neighbours] for r in batch.results]
            return recall_at_k(got, truth)

        assert recall_for(400) >= recall_for(10) - 1e-9


class TestBatchResult:
    def test_repetitions_produce_all_samples(self, dataset_id: int) -> None:
        vs.create_hnsw_index(dataset_id, DistanceMetric.L2)
        vectors = [r[2] for r in vs.load_test_vectors(dataset_id, 5)]
        batch = vs.retrieve_batch(
            vectors,
            5,
            DistanceMetric.L2,
            dataset_id,
            method="hnsw",
            ef_search=100,
            repetitions=3,
        )
        assert len(batch.results) == 5
        assert len(batch.latencies) == 15
        assert batch.repetitions == 3
        assert batch.warmup_queries == 1

    def test_repetitions_rejected_when_zero(self, dataset_id: int) -> None:
        vectors = [r[2] for r in vs.load_test_vectors(dataset_id, 2)]
        with pytest.raises(ValueError, match="repetitions"):
            vs.retrieve_batch(
                vectors, 5, DistanceMetric.L2, dataset_id, repetitions=0
            )


class TestPlanDetection:
    def test_detects_exact_index_name(self) -> None:
        plan = (
            "Index Scan using fv_1_embedding_hnsw_idx on fv_1 "
            "(cost=...) (actual time=...)\n  Buffers: shared hit=5"
        )
        assert vs.plan_uses_index_scan(plan, "fv_1_embedding_hnsw_idx")
        assert not vs.plan_uses_index_scan(plan, "some_other_index")

    def test_detects_sequential_scan(self) -> None:
        assert vs.plan_uses_sequential_scan("Seq Scan on fv_1  (cost=...)")
        assert not vs.plan_uses_sequential_scan("Index Scan using some_idx on fv_1")


class TestSchema:
    def test_hnsw_registry_table_exists(self) -> None:
        row = fetch_one(
            "SELECT count(*) AS c FROM information_schema.tables "
            "WHERE table_name = 'hnsw_indexes'"
        )
        assert int(row["c"]) == 1

    def test_experiment_measurement_columns_exist(self) -> None:
        row = fetch_one(
            "SELECT count(*) AS c FROM information_schema.columns "
            "WHERE table_name = 'experiments' "
            "AND column_name IN ('index_size_bytes','repetitions','environment',"
            "'latency_min_ms','latency_max_ms','api_overhead_s')"
        )
        assert int(row["c"]) == 6

    def test_ivfflat_column_removed(self) -> None:
        row = fetch_one(
            "SELECT count(*) AS c FROM information_schema.columns "
            "WHERE table_name = 'experiments' AND column_name = 'hnsw_lists'"
        )
        assert int(row["c"]) == 0
