"""Dataset ingestion endpoints."""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.api.schemas import (
    CovertypeIngestRequest,
    DatasetIngestRequest,
    DatasetSummary,
    IngestResponse,
)
from app.core.config import settings
from app.core.database import execute, fetch_all, fetch_one
from app.services import datasets as ds
from app.services import vector_store as vs
from app.services.preprocessing import (
    DataValidationError,
    PreprocessingMode,
    fit_preprocessor,
    transform,
    validate_dataset,
    vector_literal,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/datasets", tags=["datasets"])


def _ingest_frame(df, name: str, source: str, target_column: str, test_size: int | None,
                  max_train_rows: int | None, seed: int, original_name: str | None = None,
                  preprocessing_mode: PreprocessingMode = PreprocessingMode.COVERTYPE) -> IngestResponse:
    """Shared ingestion pipeline for uploaded and seeded datasets."""
    if target_column not in df.columns:
        raise HTTPException(400, f"Target column '{target_column}' not found in the dataset")

    # Reject unusable datasets before doing any work.
    problems = validate_dataset(df, target_column)
    if problems:
        raise HTTPException(400, "Invalid dataset: " + "; ".join(problems))

    try:
        train_df, test_df = ds.stratified_split(
            df, target_column, test_size=test_size, seed=seed
        )
    except ValueError as exc:
        raise HTTPException(400, f"Cannot split the dataset: {exc}") from exc

    train_df = ds.sample_training_rows(train_df, target_column, max_train_rows, seed=seed)

    # Preprocessing statistics come exclusively from the training split.
    try:
        state, preprocessor = fit_preprocessor(train_df, target_column, preprocessing_mode)
    except DataValidationError as exc:
        raise HTTPException(400, f"Invalid dataset: {exc}") from exc
    train_matrix, train_labels = transform(train_df, state, preprocessor)
    test_matrix, test_labels = transform(test_df, state, preprocessor)

    dimension = int(train_matrix.shape[1])
    if dimension == 0:
        raise HTTPException(400, "No feature columns were detected")

    if train_matrix.shape[1] != test_matrix.shape[1]:
        raise HTTPException(
            400,
            "The training and test splits produced different feature dimensions "
            f"({train_matrix.shape[1]} vs {test_matrix.shape[1]}).",
        )

    train_vecs = vector_literal(train_matrix)
    test_vecs = vector_literal(test_matrix)

    dataset_row = execute_returning_id(
        """
        INSERT INTO datasets
            (name, source, original_name, n_rows, n_features, n_classes, dimension,
             class_labels, preprocessing, is_seeded, preprocessing_mode)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s)
        RETURNING id
        """,
        (
            name,
            source,
            original_name,
            len(df),
            len(state.output_columns),
            len(state.class_labels),
            dimension,
            json.dumps(state.class_labels),
            state.to_json(),
            source == "covertype",
            preprocessing_mode.value,
        ),
    )

    dataset_id = dataset_row["id"]

    try:
        vs.create_dataset_tables(dataset_id, dimension)
        vs.insert_vectors(dataset_id, "train", train_vecs, train_labels,
                          train_df.index.tolist())
        vs.insert_vectors(dataset_id, "test", test_vecs, test_labels,
                          test_df.index.tolist())
    except Exception:
        execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))
        vs.drop_dataset_tables(dataset_id)
        raise

    distribution = ds.class_distribution(train_labels)

    logger.info(
        "Ingested dataset %s (%s): %s train / %s test, dim=%s",
        dataset_id, name, len(train_df), len(test_df), dimension,
    )

    return IngestResponse(
        dataset_id=dataset_id,
        name=name,
        n_train=len(train_df),
        n_test=len(test_df),
        dimension=dimension,
        n_classes=len(state.class_labels),
        classes=state.class_labels,
        class_distribution=distribution,
        preprocessing=state.to_dict(),
        preprocessing_mode=state.mode,
    )


def execute_returning_id(sql: str, params: tuple) -> dict:
    from app.core.database import connection

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchone()


@router.get("", response_model=list[DatasetSummary])
def list_datasets() -> list[DatasetSummary]:
    rows = fetch_all(
        """
        SELECT id, name, source, n_rows, n_features, n_classes, dimension,
               class_labels, preprocessing_mode, created_at
        FROM datasets ORDER BY created_at DESC
        """
    )
    return [
        DatasetSummary(
            id=r["id"],
            name=r["name"],
            source=r["source"],
            n_rows=r["n_rows"],
            n_features=r["n_features"],
            n_classes=r["n_classes"],
            dimension=r["dimension"],
            class_labels=r["class_labels"] or [],
            preprocessing_mode=r["preprocessing_mode"],
            created_at=r["created_at"].isoformat(),
        )
        for r in rows
    ]


@router.get("/{dataset_id}", response_model=DatasetSummary)
def get_dataset(dataset_id: int) -> DatasetSummary:
    row = fetch_one(
        """
        SELECT id, name, source, n_rows, n_features, n_classes, dimension,
               class_labels, preprocessing_mode, created_at
        FROM datasets WHERE id = %s
        """,
        (dataset_id,),
    )
    if row is None:
        raise HTTPException(404, "Dataset not found")
    return DatasetSummary(
        id=row["id"],
        name=row["name"],
        source=row["source"],
        n_rows=row["n_rows"],
        n_features=row["n_features"],
        n_classes=row["n_classes"],
        dimension=row["dimension"],
        class_labels=row["class_labels"] or [],
        preprocessing_mode=row["preprocessing_mode"],
        created_at=row["created_at"].isoformat(),
    )


@router.delete("/{dataset_id}")
def delete_dataset(dataset_id: int) -> dict:
    row = fetch_one("SELECT id FROM datasets WHERE id = %s", (dataset_id,))
    if row is None:
        raise HTTPException(404, "Dataset not found")
    vs.drop_dataset_tables(dataset_id)
    execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))
    return {"deleted": dataset_id}


@router.post("/upload", response_model=IngestResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: str = Form(...),
    target_column: str = Form("label"),
    test_size: int | None = Form(None),
    max_train_rows: int | None = Form(None),
    seed: int = Form(42),
    preprocessing_mode: PreprocessingMode = Form(PreprocessingMode.COVERTYPE),
) -> IngestResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Uploaded file is empty")
    try:
        df = ds.load_uploaded_csv(raw, target_column)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    return _ingest_frame(
        df, name=name, source="upload", target_column=target_column,
        test_size=test_size, max_train_rows=max_train_rows, seed=seed,
        original_name=file.filename, preprocessing_mode=preprocessing_mode,
    )


@router.post("/covertype", response_model=IngestResponse)
def ingest_covertype(request: CovertypeIngestRequest) -> IngestResponse:
    try:
        df = ds.load_covertype(max_rows=request.max_rows)
    except Exception as exc:  # network / archive problems
        logger.exception("Failed to load Covertype")
        raise HTTPException(502, f"Could not obtain the Covertype dataset: {exc}") from exc

    return _ingest_frame(
        df,
        name=request.name,
        source="covertype",
        target_column="label",
        test_size=request.test_size,
        max_train_rows=request.max_train_rows,
        seed=request.seed,
        preprocessing_mode=request.preprocessing_mode,
    )


@router.get("/covertype/status")
def covertype_status() -> dict:
    _, csv_path = ds.covertype_cache_paths()
    return {
        "cached": csv_path.exists(),
        "path": str(csv_path),
        "url": settings.covertype_url,
    }
