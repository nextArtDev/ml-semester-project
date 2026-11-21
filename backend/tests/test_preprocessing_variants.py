"""Unit tests for size-specific preprocessing helpers (no database access)."""

from __future__ import annotations

import pandas as pd
import pytest

from app.services.preprocessing import (
    DataValidationError,
    PreprocessingMode,
    fit_preprocessor,
    inverse_transform_vectors,
    labels_to_original,
    transform,
    vector_literal,
)
from scripts import run_experiments as runner


def _numeric_binary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "elevation": [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
            "slope": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "wilderness": [0, 1, 0, 1, 0, 1],
            "soil": [1, 0, 1, 0, 1, 0],
            "label": [1, 2, 3, 1, 2, 3],
        }
    )


def test_inverse_transform_recovers_numeric_and_binary_values() -> None:
    raw = _numeric_binary_frame()
    state, preprocessor = fit_preprocessor(raw, "label", PreprocessingMode.COVERTYPE)
    matrix, encoded = transform(raw, state, preprocessor)

    restored = inverse_transform_vectors(vector_literal(matrix), state)

    assert list(restored.columns) == ["elevation", "slope", "wilderness", "soil"]
    assert restored["elevation"].to_numpy() == pytest.approx(
        raw["elevation"].to_numpy(), abs=1e-3
    )
    assert restored["slope"].to_numpy() == pytest.approx(
        raw["slope"].to_numpy(), abs=1e-3
    )
    assert restored["wilderness"].tolist() == raw["wilderness"].tolist()
    assert restored["soil"].tolist() == raw["soil"].tolist()
    assert labels_to_original(encoded, state) == raw["label"].tolist()


def test_inverse_transform_rejects_categorical_blocks() -> None:
    raw = pd.DataFrame(
        {
            "distance": [1.0, 2.0, 3.0, 4.0],
            "category": ["a", "b", "a", "b"],
            "label": [1, 2, 1, 2],
        }
    )
    state, preprocessor = fit_preprocessor(raw, "label", PreprocessingMode.COVERTYPE)
    matrix, _ = transform(raw, state, preprocessor)

    with pytest.raises(DataValidationError, match="categorical"):
        inverse_transform_vectors(vector_literal(matrix), state)


def test_choose_subset_ids_is_exact_and_reproducible() -> None:
    rows = [
        {"id": index + 1, "label": index % 4}
        for index in range(100)
    ]

    first = runner._choose_subset_ids(rows, 37, seed=0)
    second = runner._choose_subset_ids(rows, 37, seed=0)

    assert len(first) == 37
    assert len(set(first)) == 37
    assert first == second
