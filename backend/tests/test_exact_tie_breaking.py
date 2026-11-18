"""Exact retrieval must be deterministic even with tied distances.

The exhaustive path orders by ``(distance, id)``; the HNSW path keeps only the
index-compatible distance ordering.  This file verifies both properties on a
synthetic dataset where many rows share exactly the same distance.
"""

from __future__ import annotations

import pytest

from app.core.database import close_pool, connection
from app.db.init_db import init_schema
from app.services import vector_store as vs
from app.services.classification import (
    evaluate_predictions,
    majority_vote,
    recall_at_k,
    tie_aware_recall_at_k,
)
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
def tied_dataset() -> int:
    """Three groups of identical vectors, 200 rows each, labels 0/1/2."""
    dim = 4
    groups = [
        ("[1,0,0,0]", 0),
        ("[0,1,0,0]", 1),
        ("[0,0,1,0]", 2),
    ]
    vectors: list[str] = []
    labels: list[int] = []
    for vec, label in groups:
        vectors.extend([vec] * 200)
        labels.extend([label] * 200)

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasets (name, source, n_rows, n_features, n_classes, "
                "dimension, class_labels, preprocessing, preprocessing_mode) "
                "VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) "
                "RETURNING id",
                (
                    "pytest-ties",
                    len(vectors),
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
    vs.insert_vectors(dataset_id, "train", vectors, labels, list(range(len(vectors))))
    vs.insert_vectors(
        dataset_id,
        "test",
        ["[1,0,0,0]"] * 10 + ["[0,1,0,0]"] * 10 + ["[0,0,1,0]"] * 10,
        [0] * 10 + [1] * 10 + [2] * 10,
        list(range(30)),
    )
    yield dataset_id
    vs.drop_dataset_tables(dataset_id)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))


class TestExactTieBreaking:
    def test_repeated_exact_retrieval_returns_the_same_ids(
        self, tied_dataset: int
    ) -> None:
        ids = []
        for _ in range(6):
            result = vs.retrieve(
                "[1,0,0,0]",
                k=5,
                metric=DistanceMetric.L2,
                dataset_id=tied_dataset,
                method="exact",
                verify_plan=False,
            )
            ids.append([n.row_id for n in result.neighbours])
        assert all(item == ids[0] for item in ids)
        assert ids[0] == sorted(ids[0])  # lowest ids win the tie

    def test_exact_plan_is_still_a_sequential_scan(self, tied_dataset: int) -> None:
        result = vs.retrieve(
            "[0,1,0,0]",
            k=5,
            metric=DistanceMetric.L2,
            dataset_id=tied_dataset,
            method="exact",
        )
        assert result.used_sequential_scan is True
        assert result.used_index_scan is False

    def test_exact_sql_has_an_id_tiebreak_but_hnsw_sql_does_not(self) -> None:
        exact = vs._query_sql(1, DistanceMetric.L2, 5, use_operator=False).as_string()
        hnsw = vs._query_sql(1, DistanceMetric.L2, 5, use_operator=True).as_string()
        assert "id" in exact.split("ORDER BY")[1]
        assert "id" not in hnsw.split("ORDER BY")[1]

    def test_predictions_are_reproducible_under_deterministic_exact(
        self, tied_dataset: int
    ) -> None:
        predictions = []
        for _ in range(3):
            result = vs.retrieve(
                "[1,0,0,0]",
                k=5,
                metric=DistanceMetric.L2,
                dataset_id=tied_dataset,
                method="exact",
                verify_plan=False,
            )
            predictions.append(
                majority_vote(
                    [n.label for n in result.neighbours],
                    [n.distance for n in result.neighbours],
                )
            )
        assert predictions == [0, 0, 0]
        metrics = evaluate_predictions([0, 0, 0], predictions, [0, 1, 2])
        assert metrics.accuracy == 1.0


class TestHnswWithTies:
    def test_hnsw_still_uses_the_index(self, tied_dataset: int) -> None:
        vs.ensure_hnsw_index(tied_dataset, DistanceMetric.L2, m=16, ef_construction=64)
        result = vs.retrieve(
            "[1,0,0,0]",
            k=5,
            metric=DistanceMetric.L2,
            dataset_id=tied_dataset,
            method="hnsw",
            ef_search=50,
        )
        assert result.used_index_scan is True
        assert result.used_sequential_scan is False

    def test_tied_hnsw_neighbours_are_not_counted_as_failures(self, tied_dataset: int) -> None:
        vs.ensure_hnsw_index(tied_dataset, DistanceMetric.L2, m=16, ef_construction=64)
        gt = vs.ground_truth_candidates(
            ["[1,0,0,0]"], 5, DistanceMetric.L2, tied_dataset
        )[0]
        assert gt.is_tied
        assert len(gt.candidate_ids) > 5

        result = vs.retrieve(
            "[1,0,0,0]",
            k=5,
            metric=DistanceMetric.L2,
            dataset_id=tied_dataset,
            method="hnsw",
            ef_search=100,
            verify_plan=False,
        )
        retrieved = [[n.row_id for n in result.neighbours]]
        conventional = recall_at_k(retrieved, [gt.ids])
        tie_aware = tie_aware_recall_at_k(retrieved, [gt.candidate_ids])
        assert tie_aware >= conventional
        # Every returned neighbour is at the boundary distance, so the
        # tie-aware metric must give full credit even if the ids differ.
        assert tie_aware == 1.0

    def test_experiment_with_ties_is_reproducible(self, tied_dataset: int) -> None:
        first = run_experiment(
            ExperimentConfig(
                dataset_id=tied_dataset,
                method="exact",
                k=5,
                max_queries=10,
                repetitions=2,
            )
        )
        second = run_experiment(
            ExperimentConfig(
                dataset_id=tied_dataset,
                method="exact",
                k=5,
                max_queries=10,
                repetitions=2,
            )
        )
        assert first["accuracy"] == second["accuracy"]
        assert first["recall_at_k"] == second["recall_at_k"]
        assert first["tie_aware_recall_at_k"] == second["tie_aware_recall_at_k"]
