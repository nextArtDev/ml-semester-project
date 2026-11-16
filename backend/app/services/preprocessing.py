"""Preprocessing pipeline.

The rules implemented here are the ones that matter for a leakage-free
experiment:

* All statistics (scaler parameters, feature means, category mappings) are
  fitted on the **training split only**.
* Categorical columns are ordinal-encoded, then scaled together with the
  numeric columns so that no feature dominates the distance computation.
* The fitted state is serialisable so the exact same transform can be
  replayed for the test split and for later experiments.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessState:
    """Fitted preprocessing state, learned on the training split only."""

    feature_columns: list[str] = field(default_factory=list)
    target_column: str = "label"
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    category_mappings: dict[str, dict[str, int]] = field(default_factory=dict)
    scaler_mean: list[float] = field(default_factory=list)
    scaler_scale: list[float] = field(default_factory=list)
    class_labels: list[int] = field(default_factory=list)
    label_to_index: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "category_mappings": self.category_mappings,
            "scaler_mean": self.scaler_mean,
            "scaler_scale": self.scaler_scale,
            "class_labels": self.class_labels,
            "label_to_index": self.label_to_index,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PreprocessState:
        state = cls(**data)
        # JSON object keys are always strings, so the label mapping has to be
        # coerced back to integers.
        state.label_to_index = {int(k): int(v) for k, v in state.label_to_index.items()}
        state.class_labels = [int(v) for v in state.class_labels]
        return state

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, raw: str) -> PreprocessState:
        return cls.from_dict(json.loads(raw))


def infer_column_types(df: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    """Split feature columns into numeric and categorical groups."""
    features = [c for c in df.columns if c != target_column]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def _label_key(value: Any) -> int:
    """Normalise a raw class label into a stable integer key."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    # Non-numeric labels are mapped deterministically by their sorted position.
    return int(hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8], 16)


def fit_preprocessor(
    df: pd.DataFrame,
    target_column: str = "label",
) -> tuple[PreprocessState, StandardScaler]:
    """Fit the preprocessing state on the given (training) dataframe."""
    numeric_cols, categorical_cols = infer_column_types(df, target_column)

    state = PreprocessState(
        feature_columns=[c for c in df.columns if c != target_column],
        target_column=target_column,
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
    )

    work = df.copy()

    for col in categorical_cols:
        categories = sorted(work[col].astype(str).unique().tolist())
        state.category_mappings[col] = {cat: idx for idx, cat in enumerate(categories)}
        work[col] = work[col].astype(str).map(state.category_mappings[col]).astype(float)

    numeric_matrix = work[state.feature_columns].astype(float).to_numpy()
    scaler = StandardScaler()
    scaler.fit(numeric_matrix)

    state.scaler_mean = [float(v) for v in scaler.mean_]
    state.scaler_scale = [float(v) for v in scaler.scale_]

    raw_labels = work[target_column].to_numpy()
    classes = sorted(pd.unique(raw_labels).tolist(), key=lambda v: (isinstance(v, str), v))
    state.class_labels = [_label_key(v) for v in classes]
    state.label_to_index = {v: i for i, v in enumerate(state.class_labels)}

    return state, scaler


def transform(
    df: pd.DataFrame,
    state: PreprocessState,
    scaler: StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fitted preprocessing, returning ``(matrix, encoded_labels)``."""
    work = df.copy()

    for col, mapping in state.category_mappings.items():
        if col not in work.columns:
            continue
        # Unknown categories map to -1, which the scaler then treats as a
        # consistent "unseen" level rather than an error.
        work[col] = work[col].astype(str).map(mapping).fillna(-1).astype(float)

    missing = [c for c in state.feature_columns if c not in work.columns]
    if missing:
        for col in missing:
            work[col] = 0.0

    matrix = work[state.feature_columns].astype(float).to_numpy()
    scaled = scaler.transform(matrix)

    labels = work[state.target_column].to_numpy()
    encoded = np.array(
        [state.label_to_index.get(_label_key(v), -1) for v in labels], dtype=np.int16
    )

    return np.ascontiguousarray(scaled, dtype=np.float32), encoded


def vector_literal(matrix: np.ndarray) -> list[str]:
    """Convert a float matrix into pgvector text literals."""
    return ["[" + ",".join(f"{v:.8g}" for v in row) + "]" for row in matrix]
