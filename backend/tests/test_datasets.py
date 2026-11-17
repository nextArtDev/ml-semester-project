"""Tests for dataset acquisition and splitting helpers."""

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
    def test_train_fraction_is_honoured(self, frame: pd.DataFrame) -> None:
        for fraction in (0.5, 0.7, 0.9):
            train, test = stratified_split(frame, "label", train_fraction=fraction)
            assert len(train) == int(round(1000 * fraction))
            assert len(test) == 1000 - len(train)

    def test_train_fraction_default_is_0_7(self, frame: pd.DataFrame) -> None:
        train, test = stratified_split(frame, "label")
        assert len(train) == 700
        assert len(test) == 300

    def test_explicit_test_size_takes_precedence(self, frame: pd.DataFrame) -> None:
        train, test = stratified_split(
            frame, "label", test_size=200, train_fraction=0.5
        )
        assert len(test) == 200
        assert len(train) == 800

    def test_class_distribution_is_preserved(self, frame: pd.DataFrame) -> None:
        train, test = stratified_split(frame, "label", test_size=300)
        for label in frame["label"].unique():
            total = int((frame["label"] == label).sum())
            in_test = int((test["label"] == label).sum())
            assert in_test / total == pytest.approx(0.3, abs=0.02)

    @pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, 1.2])
    def test_invalid_train_fraction_rejected(self, frame: pd.DataFrame, fraction: float) -> None:
        with pytest.raises(ValueError, match="train_fraction"):
            stratified_split(frame, "label", train_fraction=fraction)

    def test_invalid_test_size_rejected(self, frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="test_size"):
            stratified_split(frame, "label", test_size=0)

    @pytest.mark.parametrize("seed", [0, 42, 1234])
    def test_seed_is_reproducible(self, frame: pd.DataFrame, seed: int) -> None:
        a1, b1 = stratified_split(frame, "label", test_size=100, seed=seed)
        a2, b2 = stratified_split(frame, "label", test_size=100, seed=seed)
        pd.testing.assert_frame_equal(a1, a2)
        pd.testing.assert_frame_equal(b1, b2)

    def test_different_seeds_give_different_splits(self, frame: pd.DataFrame) -> None:
        a1, b1 = stratified_split(frame, "label", test_size=100, seed=1)
        a2, b2 = stratified_split(frame, "label", test_size=100, seed=2)
        assert not b1.equals(b2)

    def test_no_row_leakage_between_splits(self, frame: pd.DataFrame) -> None:
        train, test = stratified_split(frame, "label", test_size=100)
        train_rows = {tuple(row) for row in train.to_numpy().tolist()}
        test_rows = {tuple(row) for row in test.to_numpy().tolist()}
        assert not (train_rows & test_rows)

    def test_missing_target_column_rejected(self, frame: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="not found"):
            stratified_split(frame, "missing_target")

    def test_rare_classes_stay_in_training(self) -> None:
        df = pd.DataFrame(
            {
                "x": range(12),
                "label": [0] * 10 + [1, 2],
            }
        )
        train, test = stratified_split(df, "label", test_size=3, seed=0)
        # Classes 1 and 2 have a single member and cannot be split, so they
        # must remain entirely in the training split.
        assert 1 in train["label"].values
        assert 2 in train["label"].values
        assert 1 not in test["label"].values
        assert 2 not in test["label"].values
        assert len(test) == 3

    def test_two_rows_produces_one_test_row(self) -> None:
        df = pd.DataFrame({"x": [1, 2], "label": [0, 1]})
        train, test = stratified_split(df, "label", train_fraction=0.5)
        assert len(train) == 1
        assert len(test) == 1


class TestSampleTrainingRows:
    def test_no_sampling_when_small_enough(self, frame: pd.DataFrame) -> None:
        out = sample_training_rows(frame, "label", max_rows=5000)
        assert len(out) == len(frame)

    def test_reduces_to_requested_size(self, frame: pd.DataFrame) -> None:
        out = sample_training_rows(frame, "label", max_rows=200)
        assert len(out) <= 220
        assert len(out) < len(frame)

    def test_keeps_every_class(self, frame: pd.DataFrame) -> None:
        out = sample_training_rows(frame, "label", max_rows=200)
        assert set(out["label"].unique()) == set(frame["label"].unique())

    def test_preserves_relative_proportions(self, frame: pd.DataFrame) -> None:
        out = sample_training_rows(frame, "label", max_rows=400)
        for label in frame["label"].unique():
            assert (out["label"] == label).mean() == pytest.approx(
                (frame["label"] == label).mean(), abs=0.05
            )

    def test_sampling_is_reproducible(self, frame: pd.DataFrame) -> None:
        a = sample_training_rows(frame, "label", max_rows=200, seed=7)
        b = sample_training_rows(frame, "label", max_rows=200, seed=7)
        pd.testing.assert_frame_equal(a, b)


class TestCovertypeColumns:
    def test_column_layout_matches_the_uci_file(self) -> None:
        assert len(COVERTYPE_COLUMNS) == 55
        assert COVERTYPE_COLUMNS[-1] == "label"
        assert sum(c.startswith("Soil_Type_") for c in COVERTYPE_COLUMNS) == 40
        assert sum(c.startswith("Wilderness_Area_") for c in COVERTYPE_COLUMNS) == 4
        assert len(COVERTYPE_COLUMNS[:10]) == 10


class TestClassDistribution:
    def test_counts_per_class(self) -> None:
        assert class_distribution([0, 0, 1, 2, 2, 2]) == [2, 1, 3]
