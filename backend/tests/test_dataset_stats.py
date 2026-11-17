"""Integration tests for duplicate-vector statistics and tie detection.

A synthetic dataset is built with deliberately duplicated embeddings so the
statistics can be asserted exactly. This is where the *database-level* tie
semantics are tested: which ids land in the candidate set at the K-th
boundary, and whether the tolerance admits a near-identical vector.
"""

from __future__ import annotations

import pytest

from app.core.database import close_pool, connection
from app.db.init_db import init_schema
from app.services import vector_store as vs
from app.services.classification import recall_at_k, tie_aware_recall_at_k
from app.services.dataset_stats import (
    dataset_fingerprint,
    duplicate_vector_stats,
    tie_statistics,
)
from app.services.metrics import DistanceMetric
from app.services.vector_store import TieTolerance


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
def duplicate_dataset() -> int:
    """Ten rows: a 3-row duplicate group (mixed labels) and a 2-row group."""
    dim = 4
    embeddings = [
        [1, 0, 0, 0],  # ids 1..3 share this vector
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],  # ids 4..5 share this vector
        [0, 1, 0, 0],
        [0, 0, 1, 0],  # ids 6..10 are distinct
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 1, 0],
        [1, 0, 1, 0],
    ]
    labels = [0, 0, 1, 1, 1, 2, 2, 2, 2, 2]

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasets (name, source, n_rows, n_features, n_classes, "
                "dimension, class_labels, preprocessing, preprocessing_mode) "
                "VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) "
                "RETURNING id",
                (
                    "pytest-duplicates",
                    len(embeddings),
                    dim,
                    3,
                    dim,
                    "[0, 1, 2]",
                    '{"class_labels": [0, 1, 2], "label_to_index": {"0": 0, "1": 1, "2": 2}}',
                    "covertype",
                ),
            )
            dataset_id = int(cur.fetchone()["id"])

    vs.create_dataset_tables(dataset_id, dim)
    vs.insert_vectors(
        dataset_id,
        "train",
        ["[" + ",".join(f"{v:.6f}" for v in row) + "]" for row in embeddings],
        labels,
        list(range(len(embeddings))),
    )
    vs.insert_vectors(
        dataset_id,
        "test",
        ["[1,0,0,0]", "[0,1,0,0]"],
        [0, 1],
        [0, 1],
    )

    yield dataset_id

    vs.drop_dataset_tables(dataset_id)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))


class TestDuplicateStatistics:
    def test_counts_are_exact(self, duplicate_dataset: int) -> None:
        stats = duplicate_vector_stats(duplicate_dataset)
        assert stats["total_vectors"] == 10
        assert stats["unique_vectors"] == 7
        assert stats["duplicate_groups"] == 2
        assert stats["duplicate_rows"] == 3
        assert stats["largest_group"] == 3

    def test_mixed_label_groups_are_detected(self, duplicate_dataset: int) -> None:
        stats = duplicate_vector_stats(duplicate_dataset)
        assert stats["mixed_label_groups"] == 1
        assert stats["mixed_label_rows"] == 3
        assert stats["mixed_label_group_pct"] == pytest.approx(50.0)

    def test_duplicate_row_percentage(self, duplicate_dataset: int) -> None:
        # 3 duplicated rows out of 10 vectors.
        stats = duplicate_vector_stats(duplicate_dataset)
        assert stats["duplicate_row_pct"] == pytest.approx(30.0)


class TestFingerprint:
    def test_is_stable_across_calls(self, duplicate_dataset: int) -> None:
        assert dataset_fingerprint(duplicate_dataset) == dataset_fingerprint(
            duplicate_dataset
        )

    def test_is_a_hex_digest(self, duplicate_dataset: int) -> None:
        fingerprint = dataset_fingerprint(duplicate_dataset)
        assert len(fingerprint) == 32
        int(fingerprint, 16)

    def test_changes_when_the_pool_changes(self, duplicate_dataset: int) -> None:
        before = dataset_fingerprint(duplicate_dataset)
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    vs.sql.SQL("DELETE FROM {t} WHERE id = (SELECT max(id) FROM {t})").format(
                        t=vs.sql.Identifier(vs.train_table(duplicate_dataset))
                    )
                )
        assert dataset_fingerprint(duplicate_dataset) != before


class TestBoundaryTies:
    def test_boundary_tie_produces_extra_candidates(self, duplicate_dataset: int) -> None:
        gt = vs.ground_truth_candidates(
            ["[1,0,0,0]"], 2, DistanceMetric.L2, duplicate_dataset
        )[0]
        assert len(gt.ids) == 2
        assert gt.is_tied is True
        assert len(gt.candidate_ids) == 3
        assert gt.boundary == pytest.approx(0.0)

    def test_tie_aware_allows_the_alternative_duplicate(self, duplicate_dataset: int) -> None:
        gt = vs.ground_truth_candidates(
            ["[1,0,0,0]"], 2, DistanceMetric.L2, duplicate_dataset
        )[0]
        retrieved = [[gt.ids[0], gt.candidate_ids[-1]]]
        assert recall_at_k(retrieved, [gt.ids]) < 1.0
        assert tie_aware_recall_at_k(retrieved, [gt.candidate_ids]) == pytest.approx(1.0)

    def test_tie_statistics_count_the_query(self, duplicate_dataset: int) -> None:
        stats = tie_statistics(duplicate_dataset, k=2, metric=DistanceMetric.L2)
        assert stats["evaluated_queries"] == 2
        assert stats["tied_queries"] == 1
        assert stats["max_candidate_set"] == 3
        # The duplicate group carries labels 0, 0 and 1, so a tie-only
        # candidate can change the majority vote.
        assert stats["queries_with_label_changing_tie"] == 1

    def test_untied_query_has_no_extra_candidates(self, duplicate_dataset: int) -> None:
        # Exactly two rows share this vector, so K=2 is filled without a tie.
        gt = vs.ground_truth_candidates(
            ["[0,1,0,0]"], 2, DistanceMetric.L2, duplicate_dataset
        )[0]
        assert gt.is_tied is False
        assert len(gt.candidate_ids) == 2


class TestTolerance:
    def test_near_duplicate_within_tolerance_is_tied(self, duplicate_dataset: int) -> None:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    vs.sql.SQL(
                        "INSERT INTO {t} (row_index, label, embedding) "
                        "VALUES (%s, %s, %s::vector)"
                    ).format(t=vs.sql.Identifier(vs.train_table(duplicate_dataset))),
                    (999, 0, "[1.0000001,0,0,0]"),
                )

        loose = vs.ground_truth_candidates(
            ["[1,0,0,0]"],
            2,
            DistanceMetric.L2,
            duplicate_dataset,
            TieTolerance(relative=1e-4, absolute=1e-6),
        )[0]
        tight = vs.ground_truth_candidates(
            ["[1,0,0,0]"],
            2,
            DistanceMetric.L2,
            duplicate_dataset,
            TieTolerance(relative=0.0, absolute=0.0),
        )[0]

        assert len(loose.candidate_ids) > len(tight.candidate_ids)
        assert loose.is_tied is True
        assert tight.is_tied is True  # the exact duplicates still tie
