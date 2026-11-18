"""Training-pool subsampling: exact targets, rare classes and seed handling."""

from __future__ import annotations

import pytest

from app.core.database import close_pool, connection
from app.db.init_db import init_schema
from app.services import vector_store as vs
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


@pytest.fixture
def imbalanced_dataset() -> int:
    """1,000 rows with a heavily imbalanced class distribution."""
    labels = [0] * 600 + [1] * 300 + [2] * 80 + [3] * 20
    dim = 4
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasets (name, source, n_rows, n_features, n_classes, "
                "dimension, class_labels, preprocessing, preprocessing_mode) "
                "VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) "
                "RETURNING id",
                (
                    "pytest-subsample",
                    len(labels),
                    dim,
                    4,
                    dim,
                    "[0,1,2,3]",
                    '{"class_labels": [0,1,2,3], "label_to_index": '
                    '{"0":0,"1":1,"2":2,"3":3}}',
                    "covertype",
                ),
            )
            dataset_id = int(cur.fetchone()["id"])

    vs.create_dataset_tables(dataset_id, dim)
    vs.insert_vectors(
        dataset_id,
        "train",
        [
            "[" + ",".join([str((i + label) % 3) for _ in range(dim)]) + "]"
            for i, label in enumerate(labels)
        ],
        labels,
        list(range(len(labels))),
    )
    yield dataset_id
    vs.drop_dataset_tables(dataset_id)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))


def _train_ids(dataset_id: int) -> list[int]:
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                vs.sql.SQL("SELECT id FROM {t} ORDER BY id").format(
                    t=vs.sql.Identifier(vs.train_table(dataset_id))
                )
            )
            return [int(r["id"]) for r in cur.fetchall()]


class TestSampleSizeAllocation:
    @pytest.mark.parametrize("target", [250, 333, 777, 999])
    def test_target_is_met_exactly(self, imbalanced_dataset: int, target: int) -> None:
        actual = runner.subset_dataset(imbalanced_dataset, target, seed=0)
        assert actual == target
        assert len(_train_ids(imbalanced_dataset)) == target

    @pytest.mark.parametrize("target", [4, 5, 20, 100])
    def test_rare_classes_are_kept_when_possible(
        self, imbalanced_dataset: int, target: int
    ) -> None:
        runner.subset_dataset(imbalanced_dataset, target, seed=42)
        counts = vs.label_counts(imbalanced_dataset, "train")
        assert set(counts) == {0, 1, 2, 3}
        assert sum(counts.values()) == target

    def test_allocation_sums_to_target(self) -> None:
        counts = {0: 600, 1: 300, 2: 80, 3: 20}
        for target in (4, 10, 100, 250, 999):
            allocation = vs.largest_remainder_allocation(counts, target)
            assert sum(allocation.values()) == target
            assert all(allocation[label] <= counts[label] for label in counts)


class TestSeedHandling:
    @pytest.mark.parametrize("seed", [0, 42, 10**9, -7])
    def test_any_integer_seed_is_accepted(self, imbalanced_dataset: int, seed: int) -> None:
        actual = runner.subset_dataset(imbalanced_dataset, 250, seed=seed)
        assert actual == 250

    @pytest.mark.parametrize("seed", [0, 42, 10**9, -7])
    def test_same_seed_reproduces_the_same_pool(
        self, imbalanced_dataset: int, seed: int
    ) -> None:
        runner.subset_dataset(imbalanced_dataset, 250, seed=seed)
        first = set(_train_ids(imbalanced_dataset))

        # Restore by resetting the table from a fresh dataset is expensive;
        # instead subsample from a fresh fixture-provided dataset via a second
        # call to the runner on a recreated copy of the pool.
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    vs.sql.SQL("TRUNCATE TABLE {t}").format(
                        t=vs.sql.Identifier(vs.train_table(imbalanced_dataset))
                    )
                )
                dim = 4
                labels = [0] * 600 + [1] * 300 + [2] * 80 + [3] * 20
                vectors = [
                    "[" + ",".join([str((i + label) % 3) for _ in range(dim)]) + "]"
                    for i, label in enumerate(labels)
                ]
                with cur.copy(
                    vs.sql.SQL(
                        "COPY {t} (id, row_index, label, embedding) FROM STDIN"
                    ).format(t=vs.sql.Identifier(vs.train_table(imbalanced_dataset)))
                ) as copy:
                    for index, (vector, label) in enumerate(zip(vectors, labels)):
                        copy.write_row((index + 1, index, label, vector))

        runner.subset_dataset(imbalanced_dataset, 250, seed=seed)
        assert set(_train_ids(imbalanced_dataset)) == first

    def test_different_seeds_can_produce_different_pools(
        self, imbalanced_dataset: int
    ) -> None:
        runner.subset_dataset(imbalanced_dataset, 250, seed=0)
        first = set(_train_ids(imbalanced_dataset))

        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    vs.sql.SQL("TRUNCATE TABLE {t}").format(
                        t=vs.sql.Identifier(vs.train_table(imbalanced_dataset))
                    )
                )
                labels = [0] * 600 + [1] * 300 + [2] * 80 + [3] * 20
                vectors = [
                    "[" + ",".join([str((i + label) % 3) for _ in range(4)]) + "]"
                    for i, label in enumerate(labels)
                ]
                with cur.copy(
                    vs.sql.SQL(
                        "COPY {t} (id, row_index, label, embedding) FROM STDIN"
                    ).format(t=vs.sql.Identifier(vs.train_table(imbalanced_dataset)))
                ) as copy:
                    for index, (vector, label) in enumerate(zip(vectors, labels)):
                        copy.write_row((index + 1, index, label, vector))

        runner.subset_dataset(imbalanced_dataset, 250, seed=1)
        assert set(_train_ids(imbalanced_dataset)) != first
