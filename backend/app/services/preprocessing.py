"""Preprocessing pipeline.

The pipeline implements a **leakage-free** transformation with three
properties that matter for a distance-based classifier:

1. **Fit / transform separation.** Every statistic — the imputation values,
   the scaler parameters and the category vocabulary — is learned from the
   training split only. ``transform`` never learns anything.
2. **Type-aware encoding.** Nominal categorical features are **one-hot
   encoded**, never ordinal-encoded. An ordinal code treats an arbitrary
   label ordering as a meaningful distance, which is wrong: "red" and
   "blue" are not two units apart on a line.
3. **Reusable fitted state.** The learned state is serialisable, so exactly
   the same transformation is applied to the test split, to any later
   experiment and to any future query.

Column typing
-------------
A feature column is assigned to exactly one of three groups:

``numeric``
    Continuous numeric values (e.g. elevation). Imputed with the training
    median and standardised.
``binary``
    Numeric columns whose training values are only 0 and 1 — indicator
    variables such as the wilderness-area and soil-type flags in Forest
    CoverType. These are kept in their original 0/1 representation, because
    standardising a Bernoulli indicator distorts the geometry of the space
    without adding information.
``categorical``
    Non-numeric (nominal) columns. Imputed with the training mode and
    one-hot encoded over the training vocabulary. A category that was not
    seen during training produces an all-zero indicator block.

Missing values
--------------
Missing feature values are imputed with training-set statistics (median for
numeric columns, mode for binary and categorical columns). Rows with a
missing *target* are rejected rather than imputed, because the label cannot
be inferred.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class PreprocessingMode(StrEnum):
    """How the feature matrix is scaled.

    ``COVERTYPE``
        Standardise the continuous numeric features and leave the binary
        indicator features in their original 0/1 representation. This is the
        primary configuration for Forest CoverType, whose 44 indicator
        columns are already one-hot encoded in the source data.

    ``STANDARDIZE_ALL``
        Standardise every feature, including the binary indicators. Provided
        so that the two configurations can be compared experimentally.
    """

    COVERTYPE = "covertype"
    STANDARDIZE_ALL = "standardize_all"


class DataValidationError(ValueError):
    """Raised when a dataset fails validation."""


# ---------------------------------------------------------------------------
# Fitted state
# ---------------------------------------------------------------------------
@dataclass
class PreprocessState:
    """Everything learned during fitting. Fitted on the training split only."""

    mode: str = PreprocessingMode.COVERTYPE.value
    target_column: str = "label"

    # Source column groups.
    numeric_columns: list[str] = field(default_factory=list)
    binary_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)

    # Imputation values, learned from the training split.
    numeric_medians: dict[str, float] = field(default_factory=dict)
    binary_modes: dict[str, float] = field(default_factory=dict)
    categorical_fill: dict[str, str] = field(default_factory=dict)

    # One-hot vocabularies, learned from the training split.
    categorical_categories: dict[str, list[str]] = field(default_factory=dict)

    # Scaler parameters.
    numeric_scaler_mean: list[float] = field(default_factory=list)
    numeric_scaler_scale: list[float] = field(default_factory=list)
    binary_scaler_mean: list[float] = field(default_factory=list)
    binary_scaler_scale: list[float] = field(default_factory=list)

    # The ordered output feature names. Identical for train and test.
    output_columns: list[str] = field(default_factory=list)

    # Label encoding.
    class_labels: list[int] = field(default_factory=list)
    label_to_index: dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "target_column": self.target_column,
            "numeric_columns": self.numeric_columns,
            "binary_columns": self.binary_columns,
            "categorical_columns": self.categorical_columns,
            "numeric_medians": self.numeric_medians,
            "binary_modes": self.binary_modes,
            "categorical_fill": self.categorical_fill,
            "categorical_categories": self.categorical_categories,
            "numeric_scaler_mean": self.numeric_scaler_mean,
            "numeric_scaler_scale": self.numeric_scaler_scale,
            "binary_scaler_mean": self.binary_scaler_mean,
            "binary_scaler_scale": self.binary_scaler_scale,
            "output_columns": self.output_columns,
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

    @property
    def dimension(self) -> int:
        return len(self.output_columns)


@dataclass
class Preprocessor:
    """Applies a fitted :class:`PreprocessState` to new data."""

    state: PreprocessState
    numeric_scaler: StandardScaler | None = None
    binary_scaler: StandardScaler | None = None

    def transform(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(feature_matrix, encoded_labels)`` for the given frame."""
        state = self.state
        work = df.copy()

        # --- Numeric (continuous) ------------------------------------------
        if state.numeric_columns:
            block = work[state.numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            block = block.fillna(pd.Series(state.numeric_medians))
            numeric_matrix = block.to_numpy(dtype=np.float64)
            if self.numeric_scaler is not None:
                numeric_matrix = self.numeric_scaler.transform(numeric_matrix)
            numeric_matrix = numeric_matrix.astype(np.float32)
        else:
            numeric_matrix = np.zeros((len(work), 0), dtype=np.float32)

        # --- Binary indicators --------------------------------------------
        if state.binary_columns:
            block = work[state.binary_columns].apply(pd.to_numeric, errors="coerce")
            block = block.fillna(pd.Series(state.binary_modes))
            binary_matrix = block.to_numpy(dtype=np.float64)
            if self.binary_scaler is not None:
                binary_matrix = self.binary_scaler.transform(binary_matrix)
            binary_matrix = binary_matrix.astype(np.float32)
        else:
            binary_matrix = np.zeros((len(work), 0), dtype=np.float32)

        # --- Categorical (one-hot) ----------------------------------------
        one_hot_blocks: list[np.ndarray] = []
        for column in state.categorical_columns:
            categories = state.categorical_categories[column]
            fill = state.categorical_fill[column]
            values = work[column].astype("object").where(work[column].notna(), fill)
            values = values.astype(str)
            lookup = {category: index for index, category in enumerate(categories)}
            block = np.zeros((len(work), len(categories)), dtype=np.float32)
            if len(categories):
                codes = values.map(lookup)
                # An unseen category leaves an all-zero indicator block, which
                # is the standard "none of the known categories" encoding.
                positions = codes.notna().to_numpy()
                indices = codes.to_numpy(dtype=np.float64)
                valid = positions & ~np.isnan(indices)
                block[np.where(valid)[0], indices[valid].astype(int)] = 1.0
            one_hot_blocks.append(block)

        if one_hot_blocks:
            one_hot_matrix = np.hstack(one_hot_blocks).astype(np.float32)
        else:
            one_hot_matrix = np.zeros((len(work), 0), dtype=np.float32)

        matrix = np.ascontiguousarray(
            np.hstack([numeric_matrix, binary_matrix, one_hot_matrix]),
            dtype=np.float32,
        )

        labels = self._encode_labels(work)
        return matrix, labels

    def _encode_labels(self, work: pd.DataFrame) -> np.ndarray:
        state = self.state
        if state.target_column not in work.columns:
            raise DataValidationError(
                f"Target column '{state.target_column}' is missing from the frame"
            )
        raw = work[state.target_column].to_numpy()
        encoded = np.array(
            [state.label_to_index.get(_label_key(v), -1) for v in raw],
            dtype=np.int16,
        )
        return encoded


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def infer_column_types(
    df: pd.DataFrame, target_column: str
) -> tuple[list[str], list[str], list[str]]:
    """Split the feature columns into numeric, binary and categorical groups.

    A numeric column is *binary* when every non-null training value is 0 or
    1; such columns are indicator variables and are left unscaled in the
    :attr:`PreprocessingMode.COVERTYPE` mode.
    """
    features = [c for c in df.columns if c != target_column]
    numeric: list[str] = []
    binary: list[str] = []
    categorical: list[str] = []

    for column in features:
        series = df[column]
        if pd.api.types.is_numeric_dtype(series):
            numeric.append(column)
            values = pd.to_numeric(series, errors="coerce").dropna().unique()
            if len(values) and set(values.tolist()) <= {0, 1}:
                binary.append(column)
                numeric.remove(column)
        else:
            categorical.append(column)

    return numeric, binary, categorical


def validate_dataset(
    df: pd.DataFrame,
    target_column: str = "label",
    min_class_count: int = 1,
) -> list[str]:
    """Validate a dataset and return the list of problems found.

    An empty list means the dataset is usable. The caller decides whether to
    raise :class:`DataValidationError` or to report the problems.
    """
    problems: list[str] = []

    if df is None or len(df) == 0:
        return ["The dataset is empty."]

    if target_column not in df.columns:
        return [
            f"Target column '{target_column}' is missing. "
            f"Available columns: {list(df.columns)}"
        ]

    features = [c for c in df.columns if c != target_column]
    if not features:
        problems.append("The dataset has no feature columns.")

    # Empty feature columns carry no information.
    for column in features:
        if df[column].isna().all():
            problems.append(f"Feature column '{column}' is entirely empty.")

    # Infinite values cannot be scaled or compared meaningfully.
    for column in features:
        if pd.api.types.is_numeric_dtype(df[column]):
            numeric = pd.to_numeric(df[column], errors="coerce")
            if np.isinf(numeric.to_numpy(dtype=np.float64)).any():
                problems.append(
                    f"Feature column '{column}' contains infinite values."
                )

    # A missing target cannot be imputed.
    missing_labels = int(df[target_column].isna().sum())
    if missing_labels:
        problems.append(
            f"Target column '{target_column}' has {missing_labels} missing value(s)."
        )

    n_classes = int(df[target_column].nunique(dropna=True))
    if n_classes < 2:
        problems.append(
            f"Target column '{target_column}' has {n_classes} distinct value(s); "
            "at least 2 are required for classification."
        )

    counts = df[target_column].value_counts()
    for label, count in counts.items():
        if int(count) < min_class_count:
            problems.append(
                f"Class '{label}' has only {int(count)} sample(s); "
                f"at least {min_class_count} are required."
            )

    return problems


def require_valid_dataset(
    df: pd.DataFrame,
    target_column: str = "label",
    min_class_count: int = 1,
) -> None:
    """Raise :class:`DataValidationError` if the dataset is not usable."""
    problems = validate_dataset(df, target_column, min_class_count)
    if problems:
        raise DataValidationError("; ".join(problems))


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------
def _label_key(value: Any) -> int:
    """Normalise a raw class label into a stable integer key."""
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return int(hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8], 16)


def fit_preprocessor(
    df: pd.DataFrame,
    target_column: str = "label",
    mode: PreprocessingMode | str = PreprocessingMode.COVERTYPE,
) -> tuple[PreprocessState, Preprocessor]:
    """Fit the preprocessing state on the given (training) dataframe.

    Returns the serialisable state and a :class:`Preprocessor` that applies
    it. Nothing outside the supplied frame influences the fitted state.
    """
    mode = PreprocessingMode(mode)
    require_valid_dataset(df, target_column)

    numeric_cols, binary_cols, categorical_cols = infer_column_types(
        df, target_column
    )

    state = PreprocessState(
        mode=mode.value,
        target_column=target_column,
        numeric_columns=numeric_cols,
        binary_columns=binary_cols,
        categorical_columns=categorical_cols,
    )

    work = df.copy()

    # --- Imputation values (training statistics only) ---------------------
    for column in numeric_cols:
        values = pd.to_numeric(work[column], errors="coerce")
        median = values.median()
        state.numeric_medians[column] = (
            float(median) if pd.notna(median) else 0.0
        )

    for column in binary_cols:
        values = pd.to_numeric(work[column], errors="coerce").dropna()
        mode_value = values.mode()
        state.binary_modes[column] = (
            float(mode_value.iloc[0]) if len(mode_value) else 0.0
        )

    for column in categorical_cols:
        series = work[column]
        non_null = series.dropna().astype(str)
        if len(non_null):
            counts = non_null.value_counts()
            state.categorical_fill[column] = str(counts.index[0])
            categories = sorted(non_null.unique().tolist())
        else:
            state.categorical_fill[column] = ""
            categories = []
        state.categorical_categories[column] = categories

    # --- Scalers ----------------------------------------------------------
    numeric_scaler: StandardScaler | None = None
    binary_scaler: StandardScaler | None = None

    if numeric_cols:
        block = work[numeric_cols].apply(pd.to_numeric, errors="coerce")
        block = block.fillna(pd.Series(state.numeric_medians))
        numeric_scaler = StandardScaler()
        numeric_scaler.fit(block.to_numpy(dtype=np.float64))
        state.numeric_scaler_mean = [float(v) for v in numeric_scaler.mean_]
        state.numeric_scaler_scale = [float(v) for v in numeric_scaler.scale_]

    if binary_cols:
        block = work[binary_cols].apply(pd.to_numeric, errors="coerce")
        block = block.fillna(pd.Series(state.binary_modes))
        if mode is PreprocessingMode.STANDARDIZE_ALL:
            binary_scaler = StandardScaler()
            binary_scaler.fit(block.to_numpy(dtype=np.float64))
            state.binary_scaler_mean = [float(v) for v in binary_scaler.mean_]
            state.binary_scaler_scale = [float(v) for v in binary_scaler.scale_]

    # --- Output column names (fixed order) --------------------------------
    output_columns: list[str] = []
    output_columns.extend(numeric_cols)
    output_columns.extend(binary_cols)
    for column in categorical_cols:
        for category in state.categorical_categories[column]:
            output_columns.append(f"{column}={category}")
    state.output_columns = output_columns

    # --- Label encoding ---------------------------------------------------
    raw_labels = work[target_column].to_numpy()
    classes = sorted(
        pd.unique(raw_labels).tolist(), key=lambda v: (isinstance(v, str), str(v))
    )
    state.class_labels = [_label_key(v) for v in classes]
    state.label_to_index = {v: i for i, v in enumerate(state.class_labels)}

    preprocessor = Preprocessor(
        state=state,
        numeric_scaler=numeric_scaler,
        binary_scaler=binary_scaler,
    )
    return state, preprocessor


def transform(
    df: pd.DataFrame,
    state: PreprocessState,
    preprocessor: Preprocessor | StandardScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the fitted preprocessing.

    Accepts either a :class:`Preprocessor` or a legacy scaler argument for
    backwards compatibility.
    """
    if isinstance(preprocessor, Preprocessor):
        return preprocessor.transform(df)
    raise TypeError(
        "transform() requires a Preprocessor; the preprocessing pipeline was "
        "rewritten to use one-hot encoding and imputation."
    )


def vector_literal(matrix: np.ndarray) -> list[str]:
    """Convert a float matrix into pgvector text literals."""
    return ["[" + ",".join(f"{v:.8g}" for v in row) + "]" for row in matrix]
