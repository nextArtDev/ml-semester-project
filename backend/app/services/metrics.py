"""Distance metrics supported by the retrieval layer."""

from __future__ import annotations

from enum import StrEnum


class DistanceMetric(StrEnum):
    """Distance metrics exposed by pgvector's HNSW/IVFFlat index operators."""

    L2 = "l2"
    COSINE = "cosine"
    L1 = "l1"

    @property
    def operator(self) -> str:
        """The pgvector ordering operator used inside ``ORDER BY``."""
        return {
            DistanceMetric.L2: "<->",
            DistanceMetric.COSINE: "<=>",
            DistanceMetric.L1: "<+>",
        }[self]

    @property
    def function(self) -> str:
        """The pgvector distance function used for exact (non-indexed) search."""
        return {
            DistanceMetric.L2: "l2_distance",
            DistanceMetric.COSINE: "cosine_distance",
            DistanceMetric.L1: "l1_distance",
        }[self]

    @property
    def label(self) -> str:
        return {
            DistanceMetric.L2: "L2 (Euclidean)",
            DistanceMetric.COSINE: "Cosine",
            DistanceMetric.L1: "L1 (Manhattan)",
        }[self]
