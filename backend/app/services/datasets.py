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
from sklearn.model_selection import train_test_split

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into train/test while preserving class distribution."""
    stratify = df[target_column] if df[target_column].nunique() > 1 else None

    if test_size is None:
        train_df, test_df = train_test_split(
            df,
            test_size=train_fraction * 0 + 0.3,
            random_state=seed,
            stratify=stratify,
        )
    else:
        n_test = min(test_size, len(df) - 1)
        n_test = max(n_test, 1)
        train_df, test_df = train_test_split(
            df,
            test_size=n_test,
            random_state=seed,
            stratify=stratify,
        )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


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
