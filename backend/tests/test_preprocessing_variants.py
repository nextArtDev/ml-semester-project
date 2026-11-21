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
from app.services.reproducibility import (
    canonical_json,
    preprocessing_state_hash,
    selected_ids_provenance,
    variant_identity_hash,
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


def test_provenance_hashes_are_deterministic_and_order_independent() -> None:
    first = selected_ids_provenance([9, 2, 7])
    second = selected_ids_provenance([7, 9, 2])

    assert first == second
    assert first["count"] == 3
    assert len(first["sha256"]) == 64
    assert canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'


def test_state_hash_tracks_fitted_state() -> None:
    raw = _numeric_binary_frame()
    state, _ = fit_preprocessor(raw, "label", PreprocessingMode.COVERTYPE)

    changed = fit_preprocessor(
        raw.assign(elevation=raw["elevation"] + 10),
        "label",
        PreprocessingMode.COVERTYPE,
    )[0]

    assert preprocessing_state_hash(state) != preprocessing_state_hash(changed)


def test_variant_identity_includes_parent_fingerprint() -> None:
    base = {
        "parent_fingerprint": "a" * 32,
        "target_size": 100,
        "training_pool_seed": 0,
        "selected_id_count": 100,
        "selected_id_sha256": "b" * 64,
        "preprocessing_mode": "covertype",
        "preprocessing_state_sha256": "c" * 64,
    }
    other = {**base, "parent_fingerprint": "d" * 32}

    assert variant_identity_hash(base) != variant_identity_hash(other)


def test_legacy_helpers_require_explicit_destructive_flag() -> None:
    with pytest.raises(RuntimeError, match="allow_destructive"):
        runner.subset_dataset(1, 10, 0)
    with pytest.raises(RuntimeError, match="allow_destructive"):
        runner.restore_pool(1, [])
