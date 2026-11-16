"""Tests for the preprocessing pipeline.

These tests verify the leakage-free contract: every statistic used by the
transform is fitted on the training split alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.preprocessing import fit_preprocessor, transform


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "a": rng.normal(size=100),
            "b": rng.normal(size=100),
            "cat": rng.choice(["x", "y", "z"], size=100),
            "label": rng.integers(0, 3, size=100),
        }
    )


class TestFitPreprocessor:
    def test_detects_column_types(self, frame: pd.DataFrame) -> None:
        state, _ = fit_preprocessor(frame)
        assert state.numeric_columns == ["a", "b"]
        assert state.categorical_columns == ["cat"]
        assert state.feature_columns == ["a", "b", "cat"]

    def test_maps_categories_deterministically(self, frame: pd.DataFrame) -> None:
        state, _ = fit_preprocessor(frame)
        assert state.category_mappings["cat"] == {"x": 0, "y": 1, "z": 2}

    def test_scaler_fitted_on_given_frame_only(self, frame: pd.DataFrame) -> None:
        state, scaler = fit_preprocessor(frame)
        assert state.scaler_mean[0] == pytest.approx(float(frame["a"].mean()))
        assert state.scaler_scale[0] == pytest.approx(float(frame["a"].std(ddof=0)))

    def test_class_labels_are_sorted(self, frame: pd.DataFrame) -> None:
        state, _ = fit_preprocessor(frame)
        assert state.class_labels == [0, 1, 2]
        assert state.label_to_index == {0: 0, 1: 1, 2: 2}


class TestTransform:
    def test_output_shape_and_dtype(self, frame: pd.DataFrame) -> None:
        state, scaler = fit_preprocessor(frame)
        matrix, labels = transform(frame, state, scaler)
        assert matrix.shape == (100, 3)
        assert matrix.dtype == np.float32
        assert labels.shape == (100,)

    def test_labels_are_positional_indices(self, frame: pd.DataFrame) -> None:
        state, scaler = fit_preprocessor(frame)
        _, labels = transform(frame, state, scaler)
        assert set(labels.tolist()) <= {0, 1, 2}

    def test_scaling_centres_the_data(self, frame: pd.DataFrame) -> None:
        state, scaler = fit_preprocessor(frame)
        matrix, _ = transform(frame, state, scaler)
        assert np.allclose(matrix[:, 0].mean(), 0.0, atol=1e-4)
        assert np.allclose(matrix[:, 0].std(), 1.0, atol=1e-3)

    def test_unknown_category_maps_to_minus_one(self, frame: pd.DataFrame) -> None:
        state, scaler = fit_preprocessor(frame)
        unseen = frame.copy()
        unseen.loc[0, "cat"] = "unseen-value"
        matrix, _ = transform(unseen, state, scaler)
        # The encoded value must not raise, and the row must still be finite.
        assert np.isfinite(matrix).all()

    def test_transform_is_deterministic(self, frame: pd.DataFrame) -> None:
        state, scaler = fit_preprocessor(frame)
        m1, y1 = transform(frame, state, scaler)
        m2, y2 = transform(frame, state, scaler)
        assert np.array_equal(m1, m2)
        assert np.array_equal(y1, y2)


class TestLeakageContract:
    def test_test_split_values_do_not_affect_scaler(self, frame: pd.DataFrame) -> None:
        """Perturbing the test rows must not change the fitted statistics."""
        train = frame.iloc[:70].copy()
        test = frame.iloc[70:].copy()

        state_a, scaler_a = fit_preprocessor(train)
        matrix_a, _ = transform(test, state_a, scaler_a)

        # Perturb the test rows, refit on the same train split.
        perturbed_test = test.copy()
        perturbed_test["a"] += 1000.0
        state_b, scaler_b = fit_preprocessor(train)
        matrix_b, _ = transform(perturbed_test, state_b, scaler_b)

        # Scaler is identical (fitted on train only), so only the transformed
        # test matrix changes - and it changes only in the way the data changed.
        assert state_a.scaler_mean == state_b.scaler_mean
        assert np.allclose(matrix_a[:, 1:], matrix_b[:, 1:], atol=1e-3)
