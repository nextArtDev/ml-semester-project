"""Validate the tie tolerance for every supported distance metric.

The tolerance is ``distance <= boundary * (1 + relative) + absolute``.  These
tests check that exactly equal distances are admitted, that a configured
larger tolerance admits a genuinely close but different distance, that the
default does not merge clearly different distances, and that the absolute term
handles boundaries near zero.  The metric definitions themselves are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from app.core.database import close_pool, connection
from app.db.init_db import init_schema
from app.services import vector_store as vs
from app.services.metrics import DistanceMetric


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


def _make_dataset(vectors: Sequence[str], labels: Sequence[int], query: str) -> int:
    dim = len(vectors[0].strip("[]").split(","))
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO datasets (name, source, n_rows, n_features, n_classes, "
                "dimension, class_labels, preprocessing, preprocessing_mode) "
                "VALUES (%s, 'test', %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) "
                "RETURNING id",
                (
                    "pytest-tolerance",
                    len(vectors),
                    dim,
                    len(set(labels)),
                    dim,
                    "[0,1,2]",
                    '{"class_labels": [0,1,2], "label_to_index": {"0":0,"1":1,"2":2}}',
                    "covertype",
                ),
            )
            dataset_id = int(cur.fetchone()["id"])
    vs.create_dataset_tables(dataset_id, dim)
    vs.insert_vectors(dataset_id, "train", list(vectors), list(labels), list(range(len(vectors))))
    vs.insert_vectors(dataset_id, "test", [query], [0], [0])
    return dataset_id


def _cleanup(dataset_id: int) -> None:
    vs.drop_dataset_tables(dataset_id)
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM datasets WHERE id = %s", (dataset_id,))


# Vectors chosen so that the boundary group and the "far" row are unambiguous
# under the metric in question.
GEOMETRY: dict[str, tuple[list[str], list[int], str]] = {
    "l2": (
        ["[1,0,0,0]", "[0,1,0,0]", "[0,1,0,0]", "[0,1,0,0]", "[0,0,2,0]"],
        [0, 1, 1, 1, 2],
        "[1,0,0,0]",
    ),
    "l1": (
        ["[1,0,0,0]", "[0,1,0,0]", "[0,1,0,0]", "[0,1,0,0]", "[0,0,3,0]"],
        [0, 1, 1, 1, 2],
        "[1,0,0,0]",
    ),
    "cosine": (
        ["[1,0,0,0]", "[0,1,0,0]", "[0,1,0,0]", "[0,1,0,0]", "[-1,0,0,0]"],
        [0, 1, 1, 1, 2],
        "[1,0,0,0]",
    ),
}


def _candidates(
    vectors: list[str],
    labels: list[int],
    query: str,
    metric: DistanceMetric,
    k: int = 2,
    **tol,
):
    dataset_id = _make_dataset(vectors, labels, query)
    try:
        tolerance = vs.TieTolerance(**tol)
        return vs.ground_truth_candidates([query], k, metric, dataset_id, tolerance)[0]
    finally:
        _cleanup(dataset_id)


class TestToleranceByMetric:
    @pytest.mark.parametrize("metric_name", ["l2", "l1", "cosine"])
    def test_exact_ties_share_the_boundary(self, metric_name: str) -> None:
        vectors, labels, query = GEOMETRY[metric_name]
        gt = _candidates(vectors, labels, query, DistanceMetric(metric_name))
        assert gt.is_tied is True
        # The nearer row (distance 0) is also inside the boundary candidate
        # set, so three tied boundary rows give four candidates in total.
        assert len(gt.candidate_ids) == 4

    @pytest.mark.parametrize("metric_name", ["l2", "l1", "cosine"])
    def test_default_does_not_merge_clearly_different_distances(
        self, metric_name: str
    ) -> None:
        vectors, labels, query = GEOMETRY[metric_name]
        gt = _candidates(vectors, labels, query, DistanceMetric(metric_name))
        assert gt.candidate_ids == [1, 2, 3, 4]  # the far row (id 5) is excluded

    @pytest.mark.parametrize("metric_name", ["l2", "l1", "cosine"])
    def test_configured_tolerance_can_admit_a_close_distance(
        self, metric_name: str
    ) -> None:
        vectors, labels, query = GEOMETRY[metric_name]
        strict = _candidates(vectors, labels, query, DistanceMetric(metric_name))
        loose = _candidates(
            vectors,
            labels,
            query,
            DistanceMetric(metric_name),
            relative=5.0,
            absolute=0.0,
        )
        assert len(loose.candidate_ids) > len(strict.candidate_ids)

    @pytest.mark.parametrize("metric_name", ["l2", "l1", "cosine"])
    def test_absolute_term_handles_a_near_zero_boundary(self, metric_name: str) -> None:
        if metric_name == "cosine":
            # Cosine distance between collinear vectors is 0 or 2 exactly, so
            # a near-zero non-zero boundary cannot be constructed in float4.
            pytest.skip("cosine distance is discrete for collinear vectors")
        vectors = ["[0,0,0,0]", "[1e-8,0,0,0]", "[5,0,0,0]"]
        labels = [0, 0, 1]
        query = "[0,0,0,0]"
        gt = _candidates(vectors, labels, query, DistanceMetric(metric_name), k=1)
        # The zero-distance row is the boundary; the tiny-distance row is
        # admitted by the absolute term, the far row is not.
        assert gt.boundary == 0.0
        assert len(gt.candidate_ids) == 2

    @pytest.mark.parametrize("metric_name", ["l2", "l1"])
    def test_large_boundary_uses_the_relative_term(self, metric_name: str) -> None:
        vectors = [
            "[1000,0,0,0]",
            "[1000.002,0,0,0]",
            "[1000.002,0,0,0]",
            "[1000.003,0,0,0]",
            "[1010,0,0,0]",
        ]
        labels = [0, 1, 1, 1, 2]
        query = "[1000,0,0,0]"
        strict = _candidates(vectors, labels, query, DistanceMetric(metric_name), k=2)
        loose = _candidates(
            vectors,
            labels,
            query,
            DistanceMetric(metric_name),
            k=2,
            relative=1.0,
            absolute=0.0,
        )
        assert len(strict.candidate_ids) == 3
        assert len(loose.candidate_ids) == 4
        assert 4 in loose.candidate_ids
        assert 5 not in loose.candidate_ids

    @pytest.mark.parametrize("metric_name", ["l2", "l1", "cosine"])
    def test_more_than_k_equally_valid_candidates(self, metric_name: str) -> None:
        vectors, labels, query = GEOMETRY[metric_name]
        gt = _candidates(vectors, labels, query, DistanceMetric(metric_name), k=2)
        assert len(gt.ids) == 2
        assert len(gt.candidate_ids) >= 3
