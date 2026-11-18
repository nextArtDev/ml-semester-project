"""Tests for the batch experiment runner and the persisted-record contract.

These tests are deliberately end-to-end against the real database: the runner
mutates the training pool, builds a real HNSW index and writes real CSVs, so a
mocked version would not prove that the artifacts are consistent with the
stored experiments.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

from app.core.database import close_pool, connection, fetch_one
from app.db.init_db import init_schema
from app.services import vector_store as vs
from app.services.experiments import ExperimentConfig, run_experiment
from app.services.metrics import DistanceMetric
from scripts import run_experiments as runner


def _db_available() -> bool:
    try:
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT extname FROM pg_extension WHERE extname = 'vector'")
                return cur.fetchone() is not None
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _db_available(), reason="PostgreSQL with pgvector is not reachable"
)


@pytest.fixture(scope="module", autouse=True)
def schema() -> None:
    init_schema()
    yield
    close_pool()


def _make_dataset(name: str, n_train: int = 3_000, n_test: int = 60, dim: int = 4) -> int:
    """Deterministic synthetic dataset with three separable classes."""
    class_labels = [0, 1, 2]

    train_vectors: list[str] = []
    train_labels: list[int] = []
    for i in range(n_train):
        label = i % 3
        base = [0.0] * dim
        base[label] = 1.0
        # Small deterministic offset so rows are not byte-identical.
        base[dim - 1] = (i % 7) / 100.0
        train_vectors.append("[" + ",".join(f"{v:.6f}" for v in base) + "]")
        train_labels.append(label)

    test_vectors: list[str] = []
    test_labels: list[int] = []
    for i in range(n_test):
        label = i % 3
        base = [0.0] * dim
        base[label] = 1.0
        base[dim - 1] = (i % 11) / 200.0
        test_vectors.append("[" + ",".join(f"{v:.6f}" for v in base) + "]")
        test_labels.append(label)

    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasets (name, source, n_rows, n_features, n_classes, "
                "dimension, class_labels, preprocessing, preprocessing_mode) "
                "VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) "
                "RETURNING id",
                (
                    name,
                    n_train,
                    dim,
                    len(class_labels),
                    dim,
                    "[0, 1, 2]",
                    '{"class_labels": [0, 1, 2], "label_to_index": {"0": 0, "1": 1, "2": 2}}',
                    "covertype",
                ),
            )
            dataset_id = int(cur.fetchone()["id"])

    vs.create_dataset_tables(dataset_id, dim)
    vs.insert_vectors(
        dataset_id, "train", train_vectors, train_labels, list(range(n_train))
    )
    vs.insert_vectors(
        dataset_id, "test", test_vectors, test_labels, list(range(n_test))
    )
    return dataset_id


@pytest.fixture
def synthetic_dataset() -> int:
    dataset_id = _make_dataset("pytest-runner")
    yield dataset_id
    vs.drop_dataset_tables(dataset_id)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))


class TestRunnerContract:
    def test_fields_are_unique(self) -> None:
        assert len(runner.FIELDS) == len(set(runner.FIELDS))
        assert len(runner.REPETITION_FIELDS) == len(set(runner.REPETITION_FIELDS))
        assert len(runner.SUMMARY_FIELDS) == len(set(runner.SUMMARY_FIELDS))

    def test_fields_cover_the_required_metadata(self) -> None:
        required = {
            "dataset_id",
            "dataset_fingerprint",
            "train_size",
            "test_size",
            "unique_queries",
            "total_test_pool",
            "retrieval_method",
            "k",
            "distance_metric",
            "hnsw_m",
            "hnsw_ef_construction",
            "hnsw_ef_search",
            "repetitions",
            "measured_queries",
            "index_build_ms",
            "index_created_ms",
            "index_reused",
            "index_reused_build_ms",
            "index_size_bytes",
            "recall_at_k",
            "tie_aware_recall_at_k",
            "accuracy",
            "macro_f1",
            "env_postgresql",
            "env_pgvector",
            "env_os",
            "env_machine",
            "query_sampling",
            "query_sample_seed",
            "benchmark_round",
            "benchmark_position",
            "benchmark_seed",
            "warm_cache",
            "total_experiment_runtime_s",
            "index_preparation_s",
            "index_build_s",
            "ground_truth_computation_s",
            "retrieval_total_s",
            "retrieval_mean_per_query_ms",
            "metrics_computation_s",
            "persistence_s",
            "orchestration_overhead_s",
        }
        assert required <= set(runner.FIELDS)

    def test_flatten_returns_exactly_the_exported_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        detail = {
            "dataset_id": 1,
            "environment": {
                "measured_queries": 30,
                "preprocessing_mode": "covertype",
                "used_sequential_scan": False,
                "applied_ef_search": 50,
                "python": "3.12.1",
                "os": "Windows",
                "machine": "AMD64",
                "cpu_cores": 8,
                "postgresql": "18.0",
                "pgvector": "0.8.7",
            },
        }
        monkeypatch.setattr(runner, "get_experiment", lambda _id: detail)
        row = runner.flatten({"id": 7, "dataset_id": 1})
        assert set(row) == set(runner.FIELDS)
        assert row["measured_queries"] == 30
        assert row["env_pgvector"] == "0.8.7"


class TestExperimentConfigCompatibility:
    def test_runner_kwargs_are_accepted(self, synthetic_dataset: int) -> None:
        config = ExperimentConfig(
            dataset_id=synthetic_dataset,
            k=5,
            distance_metric=DistanceMetric.L2,
            method="hnsw",
            ef_search=25,
            hnsw_m=8,
            hnsw_ef_construction=32,
            max_queries=10,
            repetitions=2,
            seed=123,
            name="runner-config",
        )
        assert config.repetitions == 2
        assert config.seed == 123
        assert config.hnsw_ef_construction == 32

    def test_persisted_latency_rows_match_repetitions_times_queries(
        self, synthetic_dataset: int
    ) -> None:
        result = run_experiment(
            ExperimentConfig(
                dataset_id=synthetic_dataset,
                method="exact",
                max_queries=7,
                repetitions=3,
                name="latency-count",
            )
        )
        experiment_id = int(result["id"])
        latencies = fetch_one(
            "SELECT count(*) AS c FROM query_latencies WHERE experiment_id = %s",
            (experiment_id,),
        )
        repetitions = fetch_one(
            "SELECT count(*) AS c FROM experiment_repetitions WHERE experiment_id = %s",
            (experiment_id,),
        )
        assert int(latencies["c"]) == 3 * 7
        assert int(repetitions["c"]) == 3

        # Every repetition really executed the full query set.
        per_rep = fetch_one(
            "SELECT count(DISTINCT repetition) AS reps, "
            "       min(per_rep) AS min_n, max(per_rep) AS max_n "
            "FROM ("
            "  SELECT repetition, count(*) AS per_rep FROM query_latencies "
            "  WHERE experiment_id = %s GROUP BY repetition"
            ") AS grouped",
            (experiment_id,),
        )
        assert int(per_rep["reps"]) == 3
        assert int(per_rep["min_n"]) == 7
        assert int(per_rep["max_n"]) == 7


class TestMainSmoke:
    def test_main_writes_consistent_artifacts(
        self,
        synthetic_dataset: int,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        csv_path = tmp_path / "sweep.csv"
        json_path = tmp_path / "sweep.json"
        details_path = tmp_path / "repetition_details.csv"
        summary_path = tmp_path / "repetition_summary.csv"
        samples_path = tmp_path / "query_samples.json"

        argv = [
            "run_experiments.py",
            "--dataset-id",
            str(synthetic_dataset),
            "--k",
            "3",
            "--metric",
            "l2",
            "--max-queries",
            "10",
            "--ef-search",
            "10,40",
            "--repetitions",
            "2",
            "--rounds",
            "2",
            "--order-seed",
            "5",
            "--query-sampling",
            "stratified",
            "--query-sample-seed",
            "9",
            "--hnsw-m",
            "8",
            "--hnsw-ef-construction",
            "32",
            "--output",
            str(csv_path),
            "--json",
            str(json_path),
            "--repetition-details",
            str(details_path),
            "--repetition-summary",
            str(summary_path),
            "--query-samples",
            str(samples_path),
        ]
        monkeypatch.setattr(sys, "argv", argv)

        assert runner.main() == 0

        with csv_path.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        with json_path.open(encoding="utf-8") as fh:
            json_rows = json.load(fh)

        # Two paired rounds, each with one exact and two HNSW configurations.
        assert len(rows) == 6
        assert len(json_rows) == len(rows)
        assert list(rows[0]) == runner.FIELDS

        for row in rows:
            assert row["dataset_id"] == str(synthetic_dataset)
            assert row["repetitions"] == "2"
            assert row["unique_queries"] == "10"
            assert row["measured_queries"] == "20"
            assert row["total_test_pool"] == "60"
            assert row["tie_aware_recall_at_k"] != ""
            assert row["dataset_fingerprint"]
            assert row["query_sampling"] == "stratified"
            assert row["query_sample_seed"] == "9"
            assert row["benchmark_seed"] == "5"
            assert row["warm_cache"] == "True"
            assert float(row["total_experiment_runtime_s"]) > 0

        # Every round is a permutation of all requested configurations.
        for round_number in (1, 2):
            round_rows = [r for r in rows if r["benchmark_round"] == str(round_number)]
            assert sorted(
                (r["retrieval_method"], r["hnsw_ef_search"]) for r in round_rows
            ) == [("exact", ""), ("hnsw", "10"), ("hnsw", "40")]
            assert sorted(int(r["benchmark_position"]) for r in round_rows) == [1, 2, 3]

        # The exact baseline stays comparable: one per round, same ids and k.
        exact_rows = [r for r in rows if r["retrieval_method"] == "exact"]
        assert len(exact_rows) == 2
        assert {r["unique_queries"] for r in exact_rows} == {"10"}

        # HNSW rows must be genuine index scans with the requested ef_search.
        for row in rows:
            if row["retrieval_method"] != "hnsw":
                continue
            assert row["used_index_scan"] == "True"
            assert row["applied_ef_search"] in {"10", "40"}
            assert row["hnsw_m"] == "8"
            assert row["hnsw_ef_construction"] == "32"

        built = [r for r in rows if r["index_reused"] == "False"]
        reused = [r for r in rows if r["index_reused"] == "True"]
        assert len(built) == 1
        assert len(reused) == 3
        assert built[0]["index_build_ms"] != ""
        for row in reused:
            assert row["index_build_ms"] == ""
            assert row["index_reused_build_ms"] == built[0]["index_build_ms"]

        # The JSON is the same data, not a different query.
        for csv_row, json_row in zip(rows, json_rows):
            for field in runner.FIELDS:
                if json_row[field] is None:
                    assert csv_row[field] == ""
                else:
                    assert csv_row[field] == str(json_row[field])

        with details_path.open(newline="", encoding="utf-8") as fh:
            details = list(csv.DictReader(fh))
        assert len(details) == 6 * 2
        assert list(details[0]) == runner.REPETITION_FIELDS
        for detail in details:
            assert detail["unique_queries"] == "10"
            assert detail["samples"] == "10"

        with summary_path.open(newline="", encoding="utf-8") as fh:
            summary = list(csv.DictReader(fh))
        assert len(summary) == 6 * len(runner._SUMMARY_METRICS)
        assert list(summary[0]) == runner.SUMMARY_FIELDS

        samples = json.loads(samples_path.read_text(encoding="utf-8"))
        assert len(samples) == 6
        for value in samples.values():
            assert value["query_sampling"] == "stratified"
            assert len(value["query_ids"]) == 10

