"""Tests for seeded, class-stratified held-out query selection."""

from __future__ import annotations

import json

import pytest

from app.core.database import close_pool, connection
from app.db.init_db import init_schema
from app.services import vector_store as vs
from app.services.experiments import ExperimentConfig, run_experiment
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
def sampling_dataset() -> int:
    """200 test rows: common classes plus two deliberately rare ones."""
    dim = 4
    labels = [0] * 100 + [1] * 60 + [2] * 30 + [3] * 8 + [4] * 2
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasets (name, source, n_rows, n_features, n_classes, "
                "dimension, class_labels, preprocessing, preprocessing_mode) "
                "VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) "
                "RETURNING id",
                (
                    "pytest-query-sampling",
                    len(labels),
                    dim,
                    5,
                    dim,
                    "[0,1,2,3,4]",
                    '{"class_labels": [0,1,2,3,4], "label_to_index": '
                    '{"0":0,"1":1,"2":2,"3":3,"4":4}}',
                    "covertype",
                ),
            )
            dataset_id = int(cur.fetchone()["id"])

    vs.create_dataset_tables(dataset_id, dim)
    train_vec = ["[" + ",".join(["1", "0", "0", str(i % 5)]) + "]" for i in range(120)]
    train_lab = [i % 5 for i in range(120)]
    vs.insert_vectors(dataset_id, "train", train_vec, train_lab, list(range(120)))

    test_vec = ["[" + ",".join(["0", "1", "0", str(i % 7)]) + "]" for i in range(len(labels))]
    vs.insert_vectors(dataset_id, "test", test_vec, labels, list(range(len(labels))))
    yield dataset_id
    vs.drop_dataset_tables(dataset_id)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))


class TestQuerySelection:
    def test_first_mode_keeps_historical_behaviour(self, sampling_dataset: int) -> None:
        ids = vs.select_test_ids(sampling_dataset, 25, sampling="first", seed=0)
        assert ids == list(range(1, 26))

    def test_all_rows_when_limit_exceeds_pool(self, sampling_dataset: int) -> None:
        ids = vs.select_test_ids(sampling_dataset, 10_000, sampling="stratified", seed=1)
        assert len(ids) == 200

    def test_stratified_selection_is_reproducible(self, sampling_dataset: int) -> None:
        first = vs.select_test_ids(sampling_dataset, 50, "stratified", seed=11)
        second = vs.select_test_ids(sampling_dataset, 50, "stratified", seed=11)
        assert first == second
        assert len(set(first)) == 50

    def test_different_seeds_give_different_valid_sets(self, sampling_dataset: int) -> None:
        first = vs.select_test_ids(sampling_dataset, 80, "stratified", seed=11)
        second = vs.select_test_ids(sampling_dataset, 80, "stratified", seed=12)
        assert set(first) != set(second)
        assert len(first) == len(second) == 80

    def test_rare_classes_are_represented(self, sampling_dataset: int) -> None:
        ids = vs.select_test_ids(sampling_dataset, 50, "stratified", seed=3)
        rows = vs.load_test_vectors_by_ids(sampling_dataset, ids)
        labels = [label for _, label, _ in rows]
        assert set(labels) == {0, 1, 2, 3, 4}
        assert labels.count(0) >= labels.count(4)

    def test_distribution_is_close_to_the_pool(self, sampling_dataset: int) -> None:
        pool = vs.label_counts(sampling_dataset, "test")
        ids = vs.select_test_ids(sampling_dataset, 100, "stratified", seed=5)
        rows = vs.load_test_vectors_by_ids(sampling_dataset, ids)
        chosen: dict[int, int] = {}
        for _, label, _ in rows:
            chosen[label] = chosen.get(label, 0) + 1
        for label, count in pool.items():
            expected = round(100 * count / 200)
            assert abs(chosen.get(label, 0) - expected) <= 1

    def test_no_train_test_overlap(self, sampling_dataset: int) -> None:
        selected = set(vs.select_test_ids(sampling_dataset, 100, "stratified", seed=5))
        test_vectors = {
            embedding for _, _, embedding in vs.load_test_vectors_by_ids(sampling_dataset, selected)
        }
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    vs.sql.SQL("SELECT embedding::text AS embedding FROM {t}").format(
                        t=vs.sql.Identifier(vs.train_table(sampling_dataset))
                    )
                )
                train_vectors = {r["embedding"] for r in cur.fetchall()}
        assert test_vectors.isdisjoint(train_vectors)

    def test_invalid_mode_is_rejected(self, sampling_dataset: int) -> None:
        with pytest.raises(ValueError):
            vs.select_test_ids(sampling_dataset, 10, "random", seed=0)

    def test_largest_remainder_never_loses_slots(self) -> None:
        allocation = vs._largest_remainder(7, {0: 100, 1: 60, 2: 30, 3: 8, 4: 2})
        assert sum(allocation.values()) == 7
        assert max(allocation.values()) - min(allocation.values()) <= 1 or allocation[0] > 1


class TestQuerySelectionPersistence:
    def test_exact_and_hnsw_share_ids_and_seed(self, sampling_dataset: int) -> None:
        results = []
        for method, ef in (("exact", None), ("hnsw", 10)):
            config = ExperimentConfig(
                dataset_id=sampling_dataset,
                method=method,
                ef_search=ef,
                k=3,
                max_queries=60,
                repetitions=2,
                query_sampling="stratified",
                query_sample_seed=77,
            )
            results.append(run_experiment(config))

        exact_ids = results[0]["evaluated_query_ids"]
        hnsw_ids = results[1]["evaluated_query_ids"]
        if isinstance(exact_ids, str):
            exact_ids = json.loads(exact_ids)
        if isinstance(hnsw_ids, str):
            hnsw_ids = json.loads(hnsw_ids)
        assert exact_ids == hnsw_ids
        assert len(exact_ids) == 60

        for result in results:
            with connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT query_sampling, query_sample_seed, "
                        "evaluated_query_ids FROM experiments WHERE id = %s",
                        (result["id"],),
                    )
                    row = cur.fetchone()
            assert row["query_sampling"] == "stratified"
            assert row["query_sample_seed"] == 77
            assert len(row["evaluated_query_ids"]) == 60

    def test_default_keeps_first_ids_for_api_callers(self, sampling_dataset: int) -> None:
        result = run_experiment(
            ExperimentConfig(
                dataset_id=sampling_dataset,
                method="exact",
                k=3,
                max_queries=10,
                repetitions=1,
            )
        )
        ids = result["evaluated_query_ids"]
        if isinstance(ids, str):
            ids = json.loads(ids)
        assert ids == list(range(1, 11))
