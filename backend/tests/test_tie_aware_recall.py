"""Unit tests for conventional and tie-aware Recall@K.

These test the metric definitions in isolation. The database-level behaviour
(which ids actually become candidates at a tied K-th boundary) is covered by
``tests/test_dataset_stats.py``.
"""

from __future__ import annotations

import pytest

from app.services.classification import (
    recall_at_k,
    repetition_summary,
    tie_aware_recall_at_k,
)


class TestConventionalRecall:
    def test_perfect_match(self) -> None:
        assert recall_at_k([[1, 2, 3]], [[1, 2, 3]]) == pytest.approx(1.0)

    def test_partial_match(self) -> None:
        assert recall_at_k([[1, 2, 9]], [[1, 2, 3]]) == pytest.approx(2 / 3)

    def test_empty_inputs(self) -> None:
        assert recall_at_k([], []) == 0.0

    def test_ids_are_exact_not_distance_based(self) -> None:
        # A different row at the same distance still counts as a miss.
        assert recall_at_k([[7, 8, 9]], [[1, 2, 3]]) == 0.0


class TestTieAwareRecall:
    def test_no_ties_matches_conventional(self) -> None:
        retrieved = [[4, 5, 6]]
        candidates = [[1, 2, 3, 4, 5, 6]]  # no ties: exactly k candidates
        assert tie_aware_recall_at_k(retrieved, candidates) == pytest.approx(1.0)

    def test_no_ties_partial_matches_conventional(self) -> None:
        retrieved = [[4, 5, 9]]
        candidates = [[1, 2, 3, 4, 5]]
        assert tie_aware_recall_at_k(retrieved, candidates) == pytest.approx(2 / 3)

    def test_tie_exactly_at_boundary_counts_as_hit(self) -> None:
        """Returning a boundary-tied row instead of the reference row is valid."""
        conventional = recall_at_k([[1, 2, 3, 4, 6]], [[1, 2, 3, 4, 5]])
        tie_aware = tie_aware_recall_at_k(
            [[1, 2, 3, 4, 6]], [[1, 2, 3, 4, 5, 6, 7]]
        )
        assert conventional == pytest.approx(4 / 5)
        assert tie_aware == pytest.approx(1.0)

    def test_tie_before_the_boundary(self) -> None:
        """An extra candidate closer than the boundary is also valid."""
        conventional = recall_at_k([[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 6]])
        tie_aware = tie_aware_recall_at_k(
            [[1, 2, 3, 4, 5]], [[1, 2, 3, 4, 5, 6]]
        )
        assert conventional == pytest.approx(4 / 5)
        assert tie_aware == pytest.approx(1.0)

    def test_more_than_k_rows_share_the_same_distance(self) -> None:
        candidates = [list(range(1, 11))]  # 10 rows all tied at the boundary
        retrieved = [[7, 8, 9, 10, 1]]
        assert tie_aware_recall_at_k(retrieved, candidates) == pytest.approx(1.0)

    def test_returned_row_outside_the_candidate_set_is_a_miss(self) -> None:
        assert tie_aware_recall_at_k([[99, 2, 3, 4, 5]], [[1, 2, 3, 4, 5]]) == pytest.approx(
            4 / 5
        )

    def test_candidate_set_smaller_than_k_never_exceeds_one(self) -> None:
        # Defensive: a malformed reference must not produce recall > 1.
        assert tie_aware_recall_at_k([[1, 1, 1, 1, 1]], [[1]]) <= 1.0

    def test_tie_aware_is_never_below_conventional(self) -> None:
        """The candidate set is a superset of the reference ids."""
        retrieved = [[1, 2, 3, 4, 7]]
        reference = [[1, 2, 3, 4, 5]]
        candidates = [[1, 2, 3, 4, 5, 6, 7]]
        assert tie_aware_recall_at_k(retrieved, candidates) >= recall_at_k(
            retrieved, reference
        )

    def test_empty_retrieval_scores_zero(self) -> None:
        assert tie_aware_recall_at_k([[]], [[1, 2, 3]]) == 0.0

    def test_mean_over_queries(self) -> None:
        retrieved = [[1, 2, 3], [7, 8, 9]]
        candidates = [[1, 2, 3], [1, 2, 3]]
        assert tie_aware_recall_at_k(retrieved, candidates) == pytest.approx(0.5)


class TestRepetitionSummary:
    def test_single_value_has_no_interval(self) -> None:
        summary = repetition_summary([0.5])
        assert summary["mean"] == pytest.approx(0.5)
        assert summary["sd"] == 0.0
        assert summary["ci95_low"] is None
        assert summary["ci95_high"] is None

    def test_mean_median_and_spread(self) -> None:
        summary = repetition_summary([1.0, 2.0, 3.0])
        assert summary["mean"] == pytest.approx(2.0)
        assert summary["median"] == pytest.approx(2.0)
        assert summary["sd"] == pytest.approx(1.0)
        assert summary["min"] == pytest.approx(1.0)
        assert summary["max"] == pytest.approx(3.0)
        assert summary["ci95_low"] < summary["mean"] < summary["ci95_high"]

    def test_identical_values_have_zero_dispersion(self) -> None:
        summary = repetition_summary([0.9, 0.9, 0.9])
        assert summary["sd"] == pytest.approx(0.0)
        assert summary["ci95_low"] == pytest.approx(0.9)
        assert summary["ci95_high"] == pytest.approx(0.9)

    def test_empty_input(self) -> None:
        summary = repetition_summary([])
        assert summary["mean"] is None
