"""Tests for the preprocessing pipeline.

Covers numerical preprocessing, categorical preprocessing, one-hot encoding,
missing-value handling, validation, the leakage contract and consistent
train/test transformations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.preprocessing import (
    DataValidationError,
    PreprocessingMode,
    Preprocessor,
    PreprocessState,
    fit_preprocessor,
    infer_column_types,
    require_valid_dataset,
    transform,
    validate_dataset,
)


@pytest.fixture
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "continuous_a": rng.normal(size=100),
            "continuous_b": rng.normal(size=100) * 10 + 5,
            "indicator": rng.integers(0, 2, size=100),
            "nominal": rng.choice(["red", "green", "blue"], size=100),
            "label": rng.integers(0, 3, size=100),
        }
    )


class TestColumnTyping:
    def test_infers_three_groups(self, frame: pd.DataFrame) -> None:
        numeric, binary, categorical = infer_column_types(frame, "label")
        assert numeric == ["continuous_a", "continuous_b"]
        assert binary == ["indicator"]
        assert categorical == ["nominal"]

    def test_target_is_never_a_feature(self, frame: pd.DataFrame) -> None:
        numeric, binary, categorical = infer_column_types(frame, "label")
        assert "label" not in numeric + binary + categorical

    def test_all_zero_one_numeric_column_is_binary(self) -> None:
        df = pd.DataFrame({"flag": [0, 1, 0, 1], "label": [0, 1, 0, 1]})
        numeric, binary, categorical = infer_column_types(df, "label")
        assert binary == ["flag"]
        assert numeric == []
        assert categorical == []

    def test_numeric_column_with_three_values_is_continuous(self) -> None:
        df = pd.DataFrame({"x": [0, 1, 2, 3], "label": [0, 1, 0, 1]})
        numeric, binary, _ = infer_column_types(df, "label")
        assert numeric == ["x"]


class TestFitting:
    def test_scaler_uses_training_statistics(self, frame: pd.DataFrame) -> None:
        state, _ = fit_preprocessor(frame, "label")
        block = frame[state.numeric_columns]
        assert state.numeric_scaler_mean[0] == pytest.approx(float(block.iloc[:, 0].mean()))
        assert state.numeric_scaler_scale[0] == pytest.approx(
            float(block.iloc[:, 0].std(ddof=0))
        )

    def test_binary_indicators_are_not_scaled_by_default(self, frame: pd.DataFrame) -> None:
        state, preprocessor = fit_preprocessor(frame, "label", PreprocessingMode.COVERTYPE)
        matrix, _ = transform(frame, state, preprocessor)
        idx = state.output_columns.index("indicator")
        assert set(np.unique(matrix[:, idx])).issubset({0.0, 1.0})

    def test_standardize_all_scales_binary_indicators(self, frame: pd.DataFrame) -> None:
        state, preprocessor = fit_preprocessor(
            frame, "label", PreprocessingMode.STANDARDIZE_ALL
        )
        matrix, _ = transform(frame, state, preprocessor)
        idx = state.output_columns.index("indicator")
        assert float(matrix[:, idx].mean()) == pytest.approx(0.0, abs=1e-4)
        assert float(matrix[:, idx].std()) == pytest.approx(1.0, abs=1e-3)

    def test_one_hot_vocabulary_is_sorted(self, frame: pd.DataFrame) -> None:
        state, _ = fit_preprocessor(frame, "label")
        assert state.categorical_categories["nominal"] == ["blue", "green", "red"]
        assert [c for c in state.output_columns if c.startswith("nominal=")] == [
            "nominal=blue",
            "nominal=green",
            "nominal=red",
        ]

    def test_output_columns_are_deterministic(self, frame: pd.DataFrame) -> None:
        state_a, _ = fit_preprocessor(frame, "label")
        state_b, _ = fit_preprocessor(frame, "label")
        assert state_a.output_columns == state_b.output_columns

    def test_class_labels_are_sorted(self, frame: pd.DataFrame) -> None:
        state, _ = fit_preprocessor(frame, "label")
        assert state.class_labels == [0, 1, 2]
        assert state.label_to_index == {0: 0, 1: 1, 2: 2}


class TestTransform:
    def test_shape_dtype_and_finiteness(self, frame: pd.DataFrame) -> None:
        state, preprocessor = fit_preprocessor(frame, "label")
        matrix, labels = transform(frame, state, preprocessor)
        assert matrix.shape == (100, 6)  # 2 numeric + 1 binary + 3 one-hot
        assert matrix.dtype == np.float32
        assert labels.shape == (100,)
        assert np.isfinite(matrix).all()

    def test_one_hot_rows_sum_to_one(self, frame: pd.DataFrame) -> None:
        state, preprocessor = fit_preprocessor(frame, "label")
        matrix, _ = transform(frame, state, preprocessor)
        one_hot = matrix[:, 3:6]
        assert np.allclose(one_hot.sum(axis=1), 1.0)

    def test_one_hot_is_exclusive(self, frame: pd.DataFrame) -> None:
        state, preprocessor = fit_preprocessor(frame, "label")
        matrix, _ = transform(frame, state, preprocessor)
        one_hot = matrix[:, 3:6]
        assert np.all(one_hot.sum(axis=1) == 1.0)
        assert np.all((one_hot == 0) | (one_hot == 1))

    def test_one_hot_replaces_ordinal_encoding(self, frame: pd.DataFrame) -> None:
        """No single output column may contain the ordinal code 0/1/2."""
        state, preprocessor = fit_preprocessor(frame, "label")
        matrix, _ = transform(frame, state, preprocessor)
        for column in ("nominal=blue", "nominal=green", "nominal=red"):
            idx = state.output_columns.index(column)
            assert set(np.unique(matrix[:, idx])).issubset({0.0, 1.0})

    def test_numeric_scaling_centres_the_data(self, frame: pd.DataFrame) -> None:
        state, preprocessor = fit_preprocessor(frame, "label")
        matrix, _ = transform(frame, state, preprocessor)
        assert np.allclose(matrix[:, 0].mean(), 0.0, atol=1e-4)
        assert np.allclose(matrix[:, 0].std(), 1.0, atol=1e-3)

    def test_transform_is_deterministic(self, frame: pd.DataFrame) -> None:
        state, preprocessor = fit_preprocessor(frame, "label")
        m1, y1 = transform(frame, state, preprocessor)
        m2, y2 = transform(frame, state, preprocessor)
        assert np.array_equal(m1, m2)
        assert np.array_equal(y1, y2)


class TestMissingValues:
    def test_numeric_median_imputation(self, frame: pd.DataFrame) -> None:
        frame.loc[0, "continuous_a"] = np.nan
        state, preprocessor = fit_preprocessor(frame, "label")
        assert state.numeric_medians["continuous_a"] == pytest.approx(
            float(frame["continuous_a"].median(skipna=True))
        )
        matrix, _ = transform(frame, state, preprocessor)
        assert np.isfinite(matrix).all()

    def test_binary_mode_imputation(self, frame: pd.DataFrame) -> None:
        frame.loc[0, "indicator"] = np.nan
        state, preprocessor = fit_preprocessor(frame, "label")
        assert state.binary_modes["indicator"] in {0.0, 1.0}
        matrix, _ = transform(frame, state, preprocessor)
        assert np.isfinite(matrix).all()

    def test_categorical_mode_imputation(self, frame: pd.DataFrame) -> None:
        frame.loc[0, "nominal"] = None
        state, preprocessor = fit_preprocessor(frame, "label")
        assert state.categorical_fill["nominal"] in ["blue", "green", "red"]
        matrix, _ = transform(frame, state, preprocessor)
        one_hot = matrix[:, 3:6]
        assert np.allclose(one_hot.sum(axis=1), 1.0)

    def test_unseen_category_produces_all_zero_block(self, frame: pd.DataFrame) -> None:
        state, preprocessor = fit_preprocessor(frame, "label")
        frame.loc[0, "nominal"] = "a-category-never-seen-in-training"
        matrix, _ = transform(frame, state, preprocessor)
        assert np.allclose(matrix[0, 3:6], 0.0)

    def test_imputation_statistics_come_from_training_only(self, frame: pd.DataFrame) -> None:
        train = frame.iloc[:70].copy()
        test = frame.iloc[70:].copy()
        test["continuous_a"] = 999.0  # a wildly different test distribution

        state, preprocessor = fit_preprocessor(train, "label")
        matrix, _ = transform(test, state, preprocessor)
        # The scaler is unchanged, so the test values become huge, but the
        # fitted statistics must be exactly the training ones.
        assert state.numeric_scaler_mean[0] == pytest.approx(
            float(train["continuous_a"].mean())
        )
        assert np.isfinite(matrix).all()


class TestValidation:
    def test_valid_dataset_has_no_problems(self, frame: pd.DataFrame) -> None:
        assert validate_dataset(frame, "label") == []

    def test_missing_target_is_rejected(self, frame: pd.DataFrame) -> None:
        frame.loc[0, "label"] = np.nan
        problems = validate_dataset(frame, "label")
        assert any("missing value" in p for p in problems)

    def test_missing_target_column_is_rejected(self, frame: pd.DataFrame) -> None:
        frame = frame.drop(columns=["label"])
        problems = validate_dataset(frame, "label")
        assert any("missing" in p.lower() for p in problems)

    def test_empty_column_is_rejected(self, frame: pd.DataFrame) -> None:
        frame["empty"] = np.nan
        problems = validate_dataset(frame, "label")
        assert any("entirely empty" in p for p in problems)

    def test_infinite_values_are_rejected(self, frame: pd.DataFrame) -> None:
        frame.loc[0, "continuous_a"] = np.inf
        problems = validate_dataset(frame, "label")
        assert any("infinite" in p for p in problems)

    def test_single_class_is_rejected(self, frame: pd.DataFrame) -> None:
        frame["label"] = 1
        problems = validate_dataset(frame, "label")
        assert any("distinct value" in p for p in problems)

    def test_empty_dataframe_is_rejected(self) -> None:
        problems = validate_dataset(pd.DataFrame(), "label")
        assert any("empty" in p.lower() for p in problems)

    def test_require_valid_dataset_raises(self, frame: pd.DataFrame) -> None:
        frame["label"] = np.nan
        with pytest.raises(DataValidationError):
            require_valid_dataset(frame, "label")

    def test_min_class_count_is_enforced(self, frame: pd.DataFrame) -> None:
        frame["label"] = 0
        frame.loc[0, "label"] = 1
        problems = validate_dataset(frame, "label", min_class_count=2)
        assert any("only 1 sample" in p for p in problems)


class TestStateSerialisation:
    def test_roundtrip_json(self, frame: pd.DataFrame) -> None:
        state, _ = fit_preprocessor(frame, "label")
        restored = PreprocessState.from_json(state.to_json())
        assert restored.output_columns == state.output_columns
        assert restored.label_to_index == state.label_to_index
        assert restored.numeric_scaler_scale == state.numeric_scaler_scale
        assert restored.categorical_categories == state.categorical_categories

    def test_restored_state_transforms_identically(self, frame: pd.DataFrame) -> None:
        state, preprocessor = fit_preprocessor(frame, "label")
        restored = PreprocessState.from_json(state.to_json())
        # Rebuild a preprocessor from the restored state using the stored
        # scaler parameters (no refitting).
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.mean_ = np.asarray(restored.numeric_scaler_mean)
        scaler.scale_ = np.asarray(restored.numeric_scaler_scale)
        scaler.var_ = scaler.scale_**2
        scaler.n_features_in_ = len(scaler.mean_)
        scaler.n_samples_seen_ = 1
        rebuilt = Preprocessor(
            state=restored, numeric_scaler=scaler, binary_scaler=None
        )
        m1, y1 = transform(frame, state, preprocessor)
        m2, y2 = transform(frame, restored, rebuilt)
        assert np.allclose(m1, m2)
        assert np.array_equal(y1, y2)


class TestLeakageContract:
    def test_test_statistics_do_not_change_the_state(self, frame: pd.DataFrame) -> None:
        train = frame.iloc[:70].copy()
        test = frame.iloc[70:].copy()
        state_a, preprocessor_a = fit_preprocessor(train, "label")
        matrix_a, _ = transform(test, state_a, preprocessor_a)

        perturbed_test = test.copy()
        perturbed_test["continuous_a"] = 999.0
        state_b, preprocessor_b = fit_preprocessor(train, "label")
        matrix_b, _ = transform(perturbed_test, state_b, preprocessor_b)

        assert state_a.numeric_scaler_mean == state_b.numeric_scaler_mean
        assert state_a.numeric_medians == state_b.numeric_medians
        assert state_a.categorical_categories == state_b.categorical_categories
        assert np.allclose(matrix_a[:, 1:], matrix_b[:, 1:], atol=1e-3)

    def test_transform_never_fits(self, frame: pd.DataFrame) -> None:
        """Transforming new data must not mutate the fitted state."""
        state, preprocessor = fit_preprocessor(frame, "label")
        before = state.to_json()
        new = frame.copy()
        new["nominal"] = "unseen"
        transform(new, state, preprocessor)
        assert state.to_json() == before

    def test_train_and_test_shapes_match(self, frame: pd.DataFrame) -> None:
        train = frame.iloc[:70]
        test = frame.iloc[70:]
        state, preprocessor = fit_preprocessor(train, "label")
        m_train, _ = transform(train, state, preprocessor)
        m_test, _ = transform(test, state, preprocessor)
        assert m_train.shape[1] == m_test.shape[1] == state.dimension
