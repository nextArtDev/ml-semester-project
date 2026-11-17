"""Dataset-level statistics used to interpret the benchmark results.

Two families of facts are measured here rather than asserted:

1. **Dataset fingerprint** — a stable identifier of the training pool that an
   experiment actually ran against, so a result row can be tied to an exact
   pool composition (row ids, labels and a content checksum over the stored
   vector text). Without it, "50,000 rows" is not enough to reproduce a run.

2. **Duplicate-vector statistics** — Covertype is known to contain repeated
   feature vectors. Whether that repetition explains the gap between
   Recall@K and classification accuracy is an empirical question, so it is
   measured: how many vectors repeat, how large the largest group is, and how
   often a repeated vector carries more than one class label.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

from psycopg import sql

from app.core.database import connection
from app.services import vector_store as vs
from app.services.metrics import DistanceMetric
from app.services.vector_store import TieTolerance


def _train_table(dataset_id: int) -> sql.Identifier:
    return sql.Identifier(vs.train_table(dataset_id))


def dataset_fingerprint(dataset_id: int) -> str:
    """A stable hash of the training pool's composition and contents.

    Covers: row count, the ordered ``(id, label)`` sequence, and a numeric
    checksum derived from each row's canonical ``embedding::text``. Two pools
    with the same size but a different subsample therefore produce different
    fingerprints.
    """
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "SELECT count(*) AS c,"
                    "       md5(string_agg(id::text || ':' || label::text, ','"
                    "           ORDER BY id)) AS pool_hash,"
                    "       coalesce(sum(('x' || substr(md5(embedding::text), 1, 16))"
                    "           ::bit(64)::bigint), 0) AS content_sum"
                    " FROM {t}"
                ).format(t=_train_table(dataset_id))
            )
            row = cur.fetchone()

    payload = f"{row['c']}|{row['pool_hash']}|{row['content_sum']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def duplicate_vector_stats(dataset_id: int) -> dict:
    """Count repeated feature vectors and label conflicts among them."""
    with connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    "WITH groups AS ("
                    "  SELECT embedding::text AS vec,"
                    "         count(*) AS n,"
                    "         count(DISTINCT label) AS label_count"
                    "  FROM {t}"
                    "  GROUP BY embedding::text"
                    ")"
                    " SELECT"
                    "  (SELECT count(*) FROM {t}) AS total_vectors,"
                    "  (SELECT count(*) FROM groups) AS unique_vectors,"
                    "  (SELECT coalesce(max(n), 0) FROM groups) AS largest_group,"
                    "  (SELECT count(*) FROM groups WHERE n > 1) AS duplicate_groups,"
                    "  (SELECT coalesce(sum(n - 1), 0) FROM groups WHERE n > 1)"
                    "      AS duplicate_rows,"
                    "  (SELECT count(*) FROM groups WHERE label_count > 1)"
                    "      AS mixed_label_groups,"
                    "  (SELECT coalesce(sum(n), 0) FROM groups WHERE label_count > 1)"
                    "      AS mixed_label_rows"
                ).format(t=_train_table(dataset_id))
            )
            row = cur.fetchone()

    total = int(row["total_vectors"]) or 1
    groups = int(row["duplicate_groups"])

    return {
        "dataset_id": int(dataset_id),
        "total_vectors": int(row["total_vectors"]),
        "unique_vectors": int(row["unique_vectors"]),
        "duplicate_groups": groups,
        "duplicate_rows": int(row["duplicate_rows"]),
        "duplicate_row_pct": 100.0 * int(row["duplicate_rows"]) / total,
        "largest_group": int(row["largest_group"]),
        "mixed_label_groups": int(row["mixed_label_groups"]),
        "mixed_label_group_pct": 100.0 * int(row["mixed_label_groups"]) / groups if groups else 0.0,
        "mixed_label_rows": int(row["mixed_label_rows"]),
    }


def tie_statistics(
    dataset_id: int,
    k: int = 5,
    metric: DistanceMetric = DistanceMetric.L2,
    tolerance: TieTolerance | None = None,
    limit: int | None = None,
) -> dict:
    """How often the exact K-th neighbour distance is tied.

    For every query this recomputes the exhaustive reference set and inspects
    the candidate set at the K-th boundary. It counts:

    * queries with more than ``k`` equally-distant candidates;
    * the largest candidate set observed;
    * queries where a tie-only candidate carries a class label that does not
      appear among the exact top-k labels — the only situation in which
      arbitrary tie-breaking could change the majority vote.
    """
    tol = tolerance or TieTolerance()
    rows = vs.load_test_vectors(dataset_id, limit)
    vectors: Sequence[str] = [r[2] for r in rows]
    ground_truth = vs.ground_truth_candidates(vectors, k, metric, dataset_id, tol)

    tied_queries = 0
    max_candidates = 0
    label_changing = 0
    tie_only_same_label = 0

    for gt in ground_truth:
        max_candidates = max(max_candidates, len(gt.candidate_ids))
        if not gt.is_tied:
            continue
        tied_queries += 1
        top_labels = {gt.candidate_labels[i] for i in gt.ids}
        tie_only = gt.candidate_ids[len(gt.ids):]
        tie_labels = {gt.candidate_labels[i] for i in tie_only}
        if tie_labels - top_labels:
            label_changing += 1
        else:
            tie_only_same_label += 1

    evaluated = len(ground_truth)
    return {
        "dataset_id": int(dataset_id),
        "k": int(k),
        "distance_metric": metric.value,
        "evaluated_queries": evaluated,
        "tied_queries": tied_queries,
        "tied_query_pct": 100.0 * tied_queries / evaluated if evaluated else 0.0,
        "max_candidate_set": max_candidates,
        "queries_with_label_changing_tie": label_changing,
        "queries_with_same_label_tie": tie_only_same_label,
        "tolerance_relative": tol.relative,
        "tolerance_absolute": tol.absolute,
    }
