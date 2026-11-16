"""Tests for the dataset splitting helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.datasets import (
    COVERTYPE_COLUMNS,
    class_distribution,
    sample_training_rows,
    stratified_split,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    return pd.DataFrame(
        {
            "a": rng.normal(size=1000),
            "b": rng.normal(size=1000),
            "label": rng.choice([1, 2, 3, 4], size=1000, p=[0.4, 0.3, 0.2, 0.1]),
        }
    )


class TestStratifiedSplit:
    def test_split_sizes(self, frame: pd.DataFrame) -> None:
        train, test = stratified_split(frame, "label", test_size=200)
        assert len(train) == 800
        assert len(test) == 200

    def test_class_distribution_is_preserved(self, frame: pd.DataFrame) -> None:
        train, test = stratified_split(frame, "label", test_size=300)
        for label in frame["label"].unique():
            total = (frame["label"] == label).sum()
            in_test = (test["label"] == label).sum()
            assert in_test / total == pytest.approx(0.3, abs=0.02)

    def test_is_deterministic(self, frame: pd.DataFrame) -> None:
        a1, b1 = stratified_split(frame, "label", test_size=100, seed=7)
        a2, b2 = stratified_split(frame, "label", test_size=100, seed=7)
        pd.testing.assert_frame_equal(a1, a2)
        pd.testing.assert_frame_equal(b1, b2)

    def test_no_row_leakage_between_splits(self, frame: pd.DataFrame) -> None:
        """The same observation must never appear in both splits."""
        train, test = stratified_split(frame, "label", test_size=100)
        train_rows = {tuple(row) for row in train.to_numpy().tolist()}
        test_rows = {tuple(row) for row in test.to_numpy().tolist()}
        assert not (train_rows & test_rows)


class TestSampleTrainingRows:
    def test_no_sampling_when_small_enough(self, frame: pd.DataFrame) -> None:
        out = sample_training_rows(frame, "label", max_rows=5000)
        assert len(out) == len(frame)

    def test_reduces_to_requested_size(self, frame: pd.DataFrame) -> None:
        out = sample_training_rows(frame, "label", max_rows=200)
        assert len(out) <= 220  # rounding per class may overshoot slightly
        assert len(out) < len(frame)

    def test_keeps_every_class(self, frame: pd.DataFrame) -> None:
        out = sample_training_rows(frame, "label", max_rows=200)
        assert set(out["label"].unique()) == set(frame["label"].unique())

    def test_preserves_relative_proportions(self, frame: pd.DataFrame) -> None:
        out = sample_training_rows(frame, "label", max_rows=400)
        for label in frame["label"].unique():
            original = (frame["label"] == label).mean()
            sampled = (out["label"] == label).mean()
            assert sampled == pytest.approx(original, abs=0.05)


class TestCovertypeColumns:
    def test_column_layout_matches_the_uci_file(self) -> None:
        # 10 continuous + 4 wilderness + 40 soil + 1 label = 55 columns.
        assert len(COVERTYPE_COLUMNS) == 55
        assert COVERTYPE_COLUMNS[-1] == "label"
        assert sum(c.startswith("Soil_Type_") for c in COVERTYPE_COLUMNS) == 40
        assert sum(c.startswith("Wilderness_Area_") for c in COVERTYPE_COLUMNS) == 4


class TestClassDistribution:
    def test_counts_per_class(self) -> None:
        assert class_distribution([0, 0, 1, 2, 2, 2]) == [2, 1, 3]
