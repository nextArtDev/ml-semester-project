"""Unit tests for the classification and metric helpers.

These tests do not require a database connection.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.services.classification import (
    evaluate_predictions,
    latency_stats,
    majority_vote,
    recall_at_k,
)
from app.services.preprocessing import PreprocessState


class TestMajorityVote:
    def test_plain_majority(self) -> None:
        assert majority_vote([1, 1, 2, 3]) == 1
        assert majority_vote([1, 2, 3]) == 1  # tie -> lowest label wins

    def test_weighted_voting_prefers_nearer_neighbours(self) -> None:
        labels = [1, 2, 2]
        distances = [0.9, 0.1, 0.2]
        assert majority_vote(labels, distances, weighted=True) == 2
        assert majority_vote(labels, distances, weighted=False) == 2

    def test_weighted_voting_flips_on_distance(self) -> None:
        labels = [1, 1, 2]
        distances = [0.9, 0.9, 0.01]
        assert majority_vote(labels, distances, weighted=False) == 1
        assert majority_vote(labels, distances, weighted=True) == 2

    def test_empty_neighbours_raises(self) -> None:
        with pytest.raises(ValueError):
            majority_vote([])


class TestEvaluatePredictions:
    def test_perfect_predictions(self) -> None:
        y = [0, 1, 2, 0, 1, 2]
        metrics = evaluate_predictions(y, y, [10, 20, 30])
        assert metrics.accuracy == 1.0
        assert metrics.macro_f1 == 1.0
        assert metrics.confusion_matrix == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]

    def test_partial_predictions(self) -> None:
        y = [0, 0, 1, 1]
        p = [0, 1, 1, 1]
        metrics = evaluate_predictions(y, p, [0, 1])
        assert metrics.accuracy == pytest.approx(0.75)
        assert metrics.confusion_matrix == [[1, 1], [0, 2]]

    def test_labels_are_display_values_not_indices(self) -> None:
        """The label list holds original values; y/y_pred hold positions."""
        y = [0, 1, 1]
        p = [0, 1, 0]
        metrics = evaluate_predictions(y, p, [101, 202])
        assert [row["label"] for row in metrics.per_class] == [101, 202]

    def test_empty_input(self) -> None:
        metrics = evaluate_predictions([], [], [1, 2])
        assert metrics.accuracy == 0.0
        assert metrics.confusion_matrix == [[0, 0], [0, 0]]

    def test_macro_f1_matches_manual_computation(self) -> None:
        y = [0, 0, 1, 1]
        p = [0, 1, 0, 1]
        metrics = evaluate_predictions(y, p, [0, 1])
        # class 0: tp=1, fp=1, fn=1 -> p=0.5, r=0.5, f1=0.5
        # class 1: tp=1, fp=1, fn=1 -> p=0.5, r=0.5, f1=0.5
        assert metrics.precision_macro == pytest.approx(0.5)
        assert metrics.recall_macro == pytest.approx(0.5)
        assert metrics.macro_f1 == pytest.approx(0.5)


class TestRecallAtK:
    def test_perfect_recall(self) -> None:
        assert recall_at_k([[1, 2, 3]], [[1, 2, 3]]) == 1.0

    def test_partial_recall(self) -> None:
        assert recall_at_k([[1, 9]], [[1, 2]]) == 0.5

    def test_no_overlap(self) -> None:
        assert recall_at_k([[5]], [[1]]) == 0.0

    def test_empty(self) -> None:
        assert recall_at_k([], []) == 0.0

    def test_averages_across_queries(self) -> None:
        assert recall_at_k([[1], [2]], [[1], [3]]) == pytest.approx(0.5)


class TestLatencyStats:
    def test_percentiles(self) -> None:
        samples = [1.0, 2.0, 3.0, 4.0, 100.0]
        stats = latency_stats(samples)
        assert stats["avg"] == pytest.approx(22.0)
        assert stats["p50"] == pytest.approx(3.0)
        assert stats["max"] == pytest.approx(100.0)

    def test_empty(self) -> None:
        stats = latency_stats([])
        assert stats["avg"] == 0.0


class TestPreprocessState:
    def test_roundtrip_json(self) -> None:
        state = PreprocessState(
            feature_columns=["a", "b"],
            target_column="label",
            numeric_columns=["a"],
            categorical_columns=["b"],
            category_mappings={"b": {"x": 0, "y": 1}},
            scaler_mean=[1.0, 2.0],
            scaler_scale=[1.0, 0.5],
            class_labels=[1, 2],
            label_to_index={1: 0, 2: 1},
        )
        restored = PreprocessState.from_json(state.to_json())
        assert restored.feature_columns == state.feature_columns
        assert restored.label_to_index == state.label_to_index
        assert restored.scaler_scale == state.scaler_scale


class TestDistanceMetric:
    def test_operators(self) -> None:
        from app.services.metrics import DistanceMetric

        assert DistanceMetric.L2.operator == "<->"
        assert DistanceMetric.COSINE.operator == "<=>"
        assert DistanceMetric.L1.operator == "<+>"
        assert DistanceMetric.L2.function == "l2_distance"
