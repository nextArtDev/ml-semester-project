"""Dataset acquisition and preparation.

Two ingestion paths are supported:

* **Upload** - a user-supplied CSV file with a header row and a label column.
* **Seeded** - the UCI Forest CoverType dataset, downloaded once and cached
  under ``backend/data``.

In both cases the same downstream pipeline applies: stratified train/test
split, preprocessing fitted on the training split only, and vector insertion
into PostgreSQL.
"""

from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from app.core.config import settings

logger = logging.getLogger(__name__)

COVERTYPE_COLUMNS: list[str] = [
    # 10 continuous features
    "Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
    "Vertical_Distance_To_Roadways", "Horizontal_Distance_To_Fire_Points",
    "Elevation_Of_Primary_Hydrology", "Elevation_Of_Distant_Hydrology",
    # 4 wilderness-area one-hot features
    "Wilderness_Area_1", "Wilderness_Area_2", "Wilderness_Area_3",
    "Wilderness_Area_4",
    # 40 soil-type one-hot features
    *[f"Soil_Type_{i}" for i in range(1, 41)],
    # target
    "label",
]


# ---------------------------------------------------------------------------
# Covertype
# ---------------------------------------------------------------------------
def covertype_cache_paths() -> tuple[Path, Path]:
    cache = settings.ensure_data_dir()
    return cache / "covertype.zip", cache / "covertype.csv.gz"


def download_covertype(force: bool = False) -> Path:
    """Download and cache the UCI Covertype dataset. Returns the CSV path."""
    import httpx

    zip_path, csv_path = covertype_cache_paths()
    if csv_path.exists() and not force:
        logger.info("Using cached Covertype dataset at %s", csv_path)
        return csv_path
    if zip_path.exists() and not force:
        logger.info("Using cached Covertype archive at %s", zip_path)
    else:
        logger.info("Downloading Covertype from %s", settings.covertype_url)
        with httpx.stream("GET", settings.covertype_url, follow_redirects=True, timeout=300) as resp:
            resp.raise_for_status()
            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 16):
                    fh.write(chunk)

    logger.info("Extracting Covertype archive")
    with zipfile.ZipFile(zip_path) as zf:
        member = next(
            (n for n in zf.namelist() if n.endswith("covtype.data.gz")), zf.namelist()[0]
        )
        with zf.open(member) as src, open(csv_path, "wb") as dst:
            dst.write(src.read())

    logger.info("Covertype ready at %s", csv_path)
    return csv_path


def load_covertype(max_rows: int | None = None) -> pd.DataFrame:
    """Load the Covertype dataset into a DataFrame."""
    csv_path = download_covertype()
    df = pd.read_csv(csv_path, header=None, names=COVERTYPE_COLUMNS)
    if max_rows is not None and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# CSV upload
# ---------------------------------------------------------------------------
def load_uploaded_csv(raw: bytes, target_column: str) -> pd.DataFrame:
    """Parse an uploaded CSV payload."""
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    df.columns = [str(c).strip() for c in df.columns]
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Available columns: {list(df.columns)}"
        )
    return df


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------
def stratified_split(
    df: pd.DataFrame,
    target_column: str = "label",
    test_size: int | None = None,
    train_fraction: float = settings.default_train_fraction,
    seed: int = 42,
    min_class_count: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test while preserving the class distribution.

    Parameters
    ----------
    df:
        The source frame.
    target_column:
        The name of the label column.
    test_size:
        An **absolute** number of held-out rows. When given it takes
        precedence over ``train_fraction``.
    train_fraction:
        The fraction of the dataset used for **training** (not testing).
        Must lie strictly between 0 and 1. Only used when ``test_size`` is
        ``None``.
    seed:
        The random seed. The same seed always produces the same split.
    min_class_count:
        The minimum number of samples a class needs in order to be split.
        A class with fewer members cannot be divided while preserving
        proportions, so it is placed entirely in the training split.

    Returns
    -------
    (train_df, test_df)
        Both frames are reset to a zero-based index.
    """
    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. Available columns: {list(df.columns)}"
        )

    if not 0.0 < train_fraction < 1.0:
        raise ValueError(
            f"train_fraction must be strictly between 0 and 1 (got {train_fraction}). "
            "It is the fraction of rows used for TRAINING."
        )

    total = len(df)
    if total < 2:
        raise ValueError("At least 2 rows are required to build a split.")

    if test_size is None:
        n_test = int(round(total * (1.0 - train_fraction)))
    else:
        if test_size < 1:
            raise ValueError(f"test_size must be at least 1 (got {test_size}).")
        n_test = int(test_size)

    n_test = min(n_test, total - 1)
    n_test = max(n_test, 1)

    labels = df[target_column].to_numpy()
    rng = np.random.default_rng(seed)

    # Group the row positions by class.
    groups: list[np.ndarray] = []
    for _, positions in pd.Series(np.arange(total)).groupby(
        pd.Series(labels), sort=True
    ):
        groups.append(np.asarray(positions))

    # Preferentially split classes with enough members to leave a training
    # example behind. Single-member classes go to training first, and are
    # used for the test split only if the proportionally allocated classes
    # cannot fill n_test on their own.
    preferred: list[np.ndarray] = []
    fallback: list[np.ndarray] = []
    for positions in groups:
        if len(positions) >= min_class_count:
            preferred.append(positions)
        else:
            fallback.append(positions)

    if not preferred and not fallback:
        raise ValueError("The dataset has no rows to split.")

    # Allocate the test rows proportionally, using the largest-remainder
    # method so that the allocations sum to exactly n_test.
    sizes = np.array([len(p) for p in preferred], dtype=np.float64) if preferred else np.array([], dtype=np.float64)
    if sizes.sum() > 0:
        exact = sizes / sizes.sum() * n_test
        allocation = np.floor(exact).astype(int)
        allocation = np.minimum(allocation, sizes.astype(int) - 1)
    else:
        exact = np.array([], dtype=np.float64)
        allocation = np.array([], dtype=int)

    remaining = n_test - int(allocation.sum())

    # Hand out the remaining rows to the classes with the largest fractional
    # remainder, while keeping at least one training example per class.
    if remaining > 0 and len(preferred):
        order = np.argsort(-(exact - allocation))
        for index in order:
            if remaining == 0:
                break
            if allocation[index] < sizes[index] - 1:
                allocation[index] += 1
                remaining -= 1

    # Rare classes may supply a test row when no other class can.
    fallback_allocation = np.zeros(len(fallback), dtype=int)
    if remaining > 0:
        for index in range(len(fallback)):
            if remaining == 0:
                break
            fallback_allocation[index] = 1  # each rare class has a single row
            remaining -= 1

    # If still short (e.g. a two-row dataset with one row per class), allow a
    # preferred class to give up its last training row.
    if remaining > 0 and len(preferred):
        for index in range(len(preferred)):
            if remaining == 0:
                break
            if allocation[index] < sizes[index]:
                allocation[index] += 1
                remaining -= 1

    test_positions: list[np.ndarray] = []
    train_positions: list[np.ndarray] = []

    for positions, count in zip(preferred, allocation):
        shuffled = positions.copy()
        rng.shuffle(shuffled)
        count = int(count)
        test_positions.append(shuffled[:count])
        train_positions.append(shuffled[count:])

    for positions, count in zip(fallback, fallback_allocation):
        shuffled = positions.copy()
        rng.shuffle(shuffled)
        count = int(count)
        test_positions.append(shuffled[:count])
        train_positions.append(shuffled[count:])

    test_index = np.sort(np.concatenate(test_positions)) if test_positions else np.array([], dtype=int)
    train_index = np.sort(np.concatenate(train_positions)) if train_positions else np.array([], dtype=int)

    train_df = df.iloc[train_index].reset_index(drop=True)
    test_df = df.iloc[test_index].reset_index(drop=True)

    if len(train_df) == 0 or len(test_df) == 0:
        raise ValueError(
            f"The split produced an empty partition (train={len(train_df)}, "
            f"test={len(test_df)})."
        )

    if min_class_count > 1 and fallback:
        logger.info(
            "Stratified split kept %s rare class(es) preferentially in the training split",
            len(fallback),
        )

    return train_df, test_df


def sample_training_rows(
    df: pd.DataFrame,
    target_column: str = "label",
    max_rows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Stratified subsample of the training split for dataset-size experiments."""
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)

    classes = df[target_column].value_counts()
    selected_idx: list[int] = []
    for label, count in classes.items():
        group = df.index[df[target_column] == label]
        n = min(int(count), max(1, round(max_rows * count / len(df))))
        selected_idx.extend(group.to_series().sample(n=n, random_state=seed).tolist())

    selected = df.loc[sorted(set(selected_idx))] if selected_idx else df.iloc[:0]
    return selected.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def class_distribution(labels: Iterable[int]) -> list[int]:
    counts = np.bincount(np.asarray(list(labels), dtype=np.int64))
    return [int(c) for c in counts]
