"""End-to-end derived-variant lifecycle tests.

These tests are skipped unless an isolated PostgreSQL database is supplied.
They must never run against the normal experiment database.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import pandas as pd
import pytest
from psycopg import connect
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app.core import database
from app.core.config import settings
from app.db.schema import migration_statements
from app.services import vector_store as vs
from app.services.dataset_stats import dataset_fingerprint
from app.services.experiments import get_experiment
from app.services.metrics import DistanceMetric
from app.services.preprocessing import (
    PreprocessingMode,
    PreprocessState,
    fit_preprocessor,
    inverse_transform_vectors,
    labels_to_original,
    parse_vector_literal,
    preprocessor_from_state,
    transform,
    vector_literal,
)
from app.services.reproducibility import preprocessing_state_hash, selected_ids_provenance
from scripts import run_experiments as runner

TEST_URL = os.environ.get("PHASE1_TEST_DATABASE_URL", "")
ALLOW_TESTS = os.environ.get("ALLOW_PHASE1_DB_TESTS") == "1"

pytestmark = pytest.mark.skipif(
    not (ALLOW_TESTS and TEST_URL),
    reason=(
        "Set ALLOW_PHASE1_DB_TESTS=1 and PHASE1_TEST_DATABASE_URL to an isolated "
        "test database before running these destructive lifecycle tests."
    ),
)

PREFIX = "pytest-phase11-"


def _assert_isolated_database() -> None:
    with connect(TEST_URL, row_factory=dict_row, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database() AS db")
            name = str(cur.fetchone()["db"])
    assert name != settings.postgres_db, "refusing normal experiment database"
    assert name.startswith("test_"), "test database must be named test_*"


def _cleanup(pool: ConnectionPool) -> None:
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id FROM datasets WHERE name LIKE %s", (PREFIX + "%",)
            )
            ids = [int(row["id"]) for row in cur.fetchall()]
            for dataset_id in ids:
                for table in (
                    vs.train_table(dataset_id),
                    vs.test_table(dataset_id),
                ):
                    cur.execute(
                        vs.sql.SQL("DROP TABLE IF EXISTS {t} CASCADE").format(
                            t=vs.sql.Identifier(table)
                        )
                    )
            cur.execute("DELETE FROM datasets WHERE name LIKE %s", (PREFIX + "%",))


@pytest.fixture(scope="module")
def isolated_pool():
    _assert_isolated_database()
    with connect(TEST_URL, row_factory=dict_row, autocommit=True) as conn:
        with conn.cursor() as cur:
            for statement in migration_statements():
                cur.execute(statement)

    pool = ConnectionPool(
        conninfo=TEST_URL,
        kwargs={"row_factory": dict_row, "autocommit": True},
        open=True,
    )
    original_get_pool = database.get_pool
    database.get_pool = lambda: pool
    _cleanup(pool)
    try:
        yield pool
    finally:
        _cleanup(pool)
        database.get_pool = original_get_pool
        pool.close()


def _raw_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        {
            "elevation": [100.0 + i for i in range(60)],
            "slope": [1.0 + (i % 5) for i in range(60)],
            "wilderness": [i % 2 for i in range(60)],
            "soil": [(i + 1) % 2 for i in range(60)],
            "label": [1 + (i % 3) for i in range(60)],
        }
    )
    test = pd.DataFrame(
        {
            "elevation": [150.0 + i for i in range(9)],
            "slope": [1.0 + (i % 5) for i in range(9)],
            "wilderness": [i % 2 for i in range(9)],
            "soil": [(i + 1) % 2 for i in range(9)],
            "label": [1 + (i % 3) for i in range(9)],
        }
    )
    return train, test


def _make_parent(suffix: str) -> tuple[int, pd.DataFrame, pd.DataFrame, PreprocessState]:
    train, test = _raw_frames()
    state, preprocessor = fit_preprocessor(
        train, "label", PreprocessingMode.COVERTYPE
    )
    with database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO datasets
                    (name, source, n_rows, n_features, n_classes, dimension,
                     class_labels, preprocessing, preprocessing_mode)
                VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s)
                RETURNING id
                """,
                (
                    PREFIX + "parent-" + suffix,
                    len(train) + len(test),
                    len(state.output_columns),
                    len(state.class_labels),
                    state.dimension,
                    json.dumps(state.class_labels),
                    state.to_json(),
                    state.mode,
                ),
            )
            dataset_id = int(cur.fetchone()["id"])

    vs.create_dataset_tables(dataset_id, state.dimension)
    train_matrix, train_labels = transform(train, state, preprocessor)
    test_matrix, test_labels = transform(test, state, preprocessor)
    vs.insert_vectors(
        dataset_id,
        "train",
        vector_literal(train_matrix),
        train_labels,
        list(range(len(train))),
    )
    vs.insert_vectors(
        dataset_id,
        "test",
        vector_literal(test_matrix),
        test_labels,
        list(range(len(test))),
    )
    return dataset_id, train, test, state


def _table_names() -> set[str]:
    with database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename LIKE 'fv_%' "
                "UNION SELECT tablename FROM pg_tables "
                "WHERE schemaname = 'public' AND tablename LIKE 'tv_%'"
            )
            return {str(row["tablename"]) for row in cur.fetchall()}


def _derived_count() -> int:
    with database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) AS c FROM datasets "
                "WHERE source = 'derived' AND name LIKE %s",
                (PREFIX + "%",),
            )
            return int(cur.fetchone()["c"])


def test_successful_variant_creation_is_complete_and_reproducible(isolated_pool) -> None:
    parent_id, train, test, base_state = _make_parent("success")
    parent_fingerprint = dataset_fingerprint(parent_id)

    metadata, selected_ids = runner.build_preprocessed_variant(parent_id, 30, 0)

    assert metadata["actual_size"] == 30
    assert len(selected_ids) == 30
    assert metadata["parent_fingerprint"] == parent_fingerprint
    assert selected_ids_provenance(selected_ids)["sha256"] == metadata["selected_id_sha256"]
    assert metadata["derived_fingerprint"] == dataset_fingerprint(metadata["dataset_id"])

    with database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                vs.sql.SQL(
                    "SELECT id, label, embedding::text AS embedding FROM {t} "
                    "WHERE id = ANY(%s) ORDER BY id"
                ).format(t=vs.sql.Identifier(vs.train_table(parent_id))),
                (selected_ids,),
            )
            selected_rows = cur.fetchall()

    subset_train = inverse_transform_vectors(
        [row["embedding"] for row in selected_rows], base_state
    )
    subset_train[base_state.target_column] = labels_to_original(
        [int(row["label"]) for row in selected_rows], base_state
    )
    expected_state, _ = fit_preprocessor(
        subset_train, "label", PreprocessingMode.COVERTYPE
    )
    assert preprocessing_state_hash(expected_state) == metadata["preprocessing_state_sha256"]
    assert vs.count_vectors(metadata["dataset_id"], "train") == 30
    assert vs.count_vectors(metadata["dataset_id"], "test") == len(test)

    with database.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT preprocessing FROM datasets WHERE id = %s",
                (metadata["dataset_id"],),
            )
            stored_state = PreprocessState.from_dict(cur.fetchone()["preprocessing"])
    assert preprocessing_state_hash(stored_state) == metadata["preprocessing_state_sha256"]

    _, encoded_test = transform(test, stored_state, preprocessor_from_state(stored_state))
    stored_test_vectors = [
        row[2] for row in vs.load_test_vectors(metadata["dataset_id"], None)
    ]
    expected_test_vectors = vector_literal(
        transform(test, stored_state, preprocessor_from_state(stored_state))[0]
    )
    stored_values = [parse_vector_literal(value) for value in stored_test_vectors]
    expected_values = [parse_vector_literal(value) for value in expected_test_vectors]
    assert len(stored_values) == len(expected_values)
    for stored_vector, expected_vector in zip(stored_values, expected_values):
        assert stored_vector == pytest.approx(expected_vector, abs=1e-5)
    assert len(encoded_test) == len(test)


@pytest.mark.parametrize(
    "stage", ["after_insert", "after_first_table", "during_copy"]
)
def test_failures_roll_back_without_touching_parent(isolated_pool, monkeypatch, stage) -> None:
    parent_id, _, _, _ = _make_parent("fail-" + stage)
    parent_fingerprint = dataset_fingerprint(parent_id)
    parent_tables = _table_names()
    parent_train = vs.count_vectors(parent_id, "train")
    parent_test = vs.count_vectors(parent_id, "test")
    derived_before = _derived_count()

    original_insert = runner._insert_variant_dataset_row
    original_copy = runner._copy_variant_vectors

    if stage == "after_insert":
        def inject(cur, parent, state, name, size):
            original_insert(cur, parent, state, name, size)
            raise RuntimeError("injected after dataset insertion")
        monkeypatch.setattr(runner, "_insert_variant_dataset_row", inject)
    elif stage == "after_first_table":
        def inject(cur, dataset_id, dimension):
            runner._create_variant_table(cur, dataset_id, "train", dimension)
            raise RuntimeError("injected after first vector table")
        monkeypatch.setattr(runner, "_create_variant_tables", inject)
    else:
        calls = {"n": 0}

        def inject(cur, dataset_id, which, embeddings, labels, row_indices):
            if calls["n"] == 0:
                calls["n"] += 1
                original_copy(
                    cur, dataset_id, which, embeddings[:1], labels[:1], row_indices[:1]
                )
                raise RuntimeError("injected during vector insertion")
            return original_copy(cur, dataset_id, which, embeddings, labels, row_indices)
        monkeypatch.setattr(runner, "_copy_variant_vectors", inject)

    with pytest.raises(RuntimeError, match="injected"):
        runner.build_preprocessed_variant(parent_id, 20, 0)

    assert _derived_count() == derived_before
    assert _table_names() == parent_tables
    assert dataset_fingerprint(parent_id) == parent_fingerprint
    assert vs.count_vectors(parent_id, "train") == parent_train
    assert vs.count_vectors(parent_id, "test") == parent_test


def test_success_variant_is_consumed_by_experiment_path(isolated_pool) -> None:
    parent_id, _, _, _ = _make_parent("consume")
    metadata, _ = runner.build_preprocessed_variant(parent_id, 20, 0)
    meta_without_ids = {k: v for k, v in metadata.items() if k != "selected_ids"}

    args = argparse.Namespace(
        dataset_id=parent_id,
        k=5,
        hnsw_m=16,
        hnsw_ef_construction=64,
        max_queries=5,
        repetitions=1,
        seed=0,
        query_sampling="first",
        query_sample_seed=0,
        order_seed=0,
        run_id="pytest-phase11-consume",
    )
    result = runner.run_one(
        args,
        DistanceMetric.L2,
        "exact",
        None,
        "20",
        1,
        1,
        json.dumps(["exact:-"]),
        dataset_id=metadata["dataset_id"],
        extra={"pool_variant": meta_without_ids},
    )

    assert result["dataset_id"] == metadata["dataset_id"]
    assert result["status"] == "completed"


def test_exact_selected_ids_are_not_embedded_in_experiment_config(isolated_pool, tmp_path) -> None:
    parent_id, _, _, _ = _make_parent("provenance")
    orders_path = tmp_path / "benchmark_orders.json"
    variants_path = tmp_path / "variant_ids.json"
    argv = [
        "run_experiments.py",
        "--dataset-id", str(parent_id),
        "--sizes", "20",
        "--ef-search", "10",
        "--k", "5",
        "--metric", "l2",
        "--max-queries", "5",
        "--repetitions", "1",
        "--rounds", "1",
        "--query-sampling", "first",
        "--benchmark-orders", str(orders_path),
        "--variant-ids", str(variants_path),
        "--run-id", "pytest-phase11-provenance",
    ]
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("sys.argv", argv)
    try:
        assert runner.main() == 0
    finally:
        monkeypatch.undo()

    orders = json.loads(orders_path.read_text(encoding="utf-8"))
    variants = json.loads(variants_path.read_text(encoding="utf-8"))["variants"]
    first_order = next(iter(orders.values()))

    assert first_order["selected_id_count"] == 20
    assert len(first_order["selected_id_sha256"]) == 64
    assert len(first_order["preprocessing_state_sha256"]) == 64
    assert len(variants) == 1
    variant = next(iter(variants.values()))
    assert len(variant["selected_ids"]) == 20

    experiment_id = first_order["experiment_ids"][0]
    detail: dict[str, Any] = get_experiment(experiment_id)
    config_extra = detail["config"]["extra"]["pool_variant"]
    assert "selected_ids" not in config_extra
    assert config_extra["selected_id_sha256"] == first_order["selected_id_sha256"]
