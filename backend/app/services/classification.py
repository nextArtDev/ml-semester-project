"""KNN classification and evaluation metrics.

The classifier itself is intentionally simple - majority voting over the
retrieved neighbours - because the research question is about *retrieval*,
not about the classification rule. Both the exact and the approximate
pipeline feed the same voting function so that any difference in accuracy
is attributable purely to the retrieval stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def majority_vote(labels: Sequence[int], distances: Sequence[float] | None = None,
                  weighted: bool = False) -> int:
    """Predict a class label from the retrieved neighbours.

    When ``weighted`` is enabled, each neighbour votes with a weight of
    ``1 / (distance + epsilon)`` so that closer neighbours dominate.
    """
    if not labels:
        raise ValueError("Cannot vote on an empty neighbour set")

    if weighted and distances is not None:
        eps = 1e-9
        weights = [1.0 / (float(d) + eps) for d in distances]
    else:
        weights = [1.0] * len(labels)

    tally: dict[int, float] = {}
    for label, weight in zip(labels, weights):
        tally[label] = tally.get(label, 0.0) + weight

    best = max(tally.items(), key=lambda kv: (kv[1], -kv[0]))
    return int(best[0])


@dataclass
class ClassificationMetrics:
    """Aggregate metrics for one experiment run."""

    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    macro_f1: float = 0.0
    confusion_matrix: list[list[int]] = field(default_factory=list)
    per_class: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "macro_f1": self.macro_f1,
            "confusion_matrix": self.confusion_matrix,
            "per_class": self.per_class,
        }


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_labels: Sequence[int],
) -> ClassificationMetrics:
    """Compute accuracy, macro precision/recall/F1 and a confusion matrix.

    ``y_true`` / ``y_pred`` are *positional* class indices (``0..n-1``) as
    produced by the preprocessing encoder, while ``class_labels`` holds the
    original label values used for display.
    """
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    labels = list(class_labels)
    n = len(labels)

    matrix = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true_arr, y_pred_arr):
        ti, pi = int(t), int(p)
        if not (0 <= ti < n and 0 <= pi < n):
            continue
        matrix[ti][pi] += 1

    correct = int(np.trace(matrix))
    total = int(matrix.sum())
    accuracy = correct / total if total else 0.0

    per_class: list[dict] = []
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for i, label in enumerate(labels):
        tp = int(matrix[i][i])
        fp = int(matrix[:, i].sum() - tp)
        fn = int(matrix[i].sum() - tp)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        support = int(matrix[i].sum())
        per_class.append(
            {
                "label": labels[i],
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return ClassificationMetrics(
        accuracy=accuracy,
        precision_macro=float(np.mean(precisions)) if precisions else 0.0,
        recall_macro=float(np.mean(recalls)) if recalls else 0.0,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        confusion_matrix=matrix.tolist(),
        per_class=per_class,
    )


def recall_at_k(retrieved: Sequence[Sequence[int]], ground_truth: Sequence[Sequence[int]]) -> float:
    """Conventional Recall@K: fraction of exact neighbour ids retrieved.

    The reference ids come from the exhaustive query's ``ORDER BY distance
    LIMIT k``. When several training rows share the K-th distance, PostgreSQL
    returns an arbitrary subset of them, so this metric treats an equally
    distant but differently-identified neighbour as a miss. See
    :func:`tie_aware_recall_at_k` for the tie-tolerant companion metric.
    """
    if not retrieved or not ground_truth:
        return 0.0
    scores: list[float] = []
    for got, want in zip(retrieved, ground_truth):
        if not want:
            scores.append(1.0)
            continue
        got_set = set(got)
        hits = sum(1 for w in want if w in got_set)
        scores.append(hits / len(want))
    return float(np.mean(scores))


def tie_aware_recall_at_k(
    retrieved: Sequence[Sequence[int]],
    candidate_sets: Sequence[Sequence[int]],
) -> float:
    """Tie-aware Recall@K.

    For each query, ``candidate_sets[i]`` is every training id whose distance
    is less than or equal to the exact K-th neighbour distance plus a
    documented tolerance (see :func:`app.services.vector_store.ground_truth_candidates`).
    Any retrieved neighbour inside that set is a valid answer, because an
    exhaustive scan could legitimately have returned it in place of a
    boundary-tied row.

    The score is ``|retrieved INTERSECT candidates| / k``, capped at 1.0.
    Dividing by ``k`` (rather than by the candidate-set size) keeps the
    metric comparable with conventional Recall@K: a system that returns ``k``
    valid neighbours scores 1.0 even when the candidate set is larger than
    ``k``.

    This metric is **not** a replacement for conventional Recall@K. It
    isolates the effect of arbitrary tie-breaking; the conventional metric
    still reports the identifier-level agreement.
    """
    if not retrieved or not candidate_sets:
        return 0.0

    scores: list[float] = []
    for got, candidates in zip(retrieved, candidate_sets):
        if not got:
            scores.append(0.0)
            continue
        valid = set(candidates)
        hits = sum(1 for row_id in got if row_id in valid)
        scores.append(min(hits, len(got)) / len(got))
    return float(np.mean(scores))


# Two-sided 97.5 % critical values of Student's t, indexed by degrees of
# freedom. Kept inline so the metric layer has no SciPy dependency.
_T_CRITICAL_975: dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042,
}


def _t_critical(df: int) -> float:
    if df < 1:
        return float("nan")
    if df in _T_CRITICAL_975:
        return _T_CRITICAL_975[df]
    if df < 15:
        return _T_CRITICAL_975[df - 1] if df - 1 in _T_CRITICAL_975 else 1.96
    return 1.96


def repetition_summary(values: Sequence[float]) -> dict[str, float | None]:
    """Summarise a metric measured once per repetition.

    Repetitions of the same query set are **not** independent experiments:
    they reuse the same pool, the same queries and (for HNSW) the same graph,
    so the spread measures run-to-run timing/state variability, not sampling
    error over a population. The returned standard deviation and confidence
    interval must be read that way.
    """
    clean = [float(v) for v in values if v is not None]
    if not clean:
        return {"mean": None, "median": None, "sd": None, "min": None,
                "max": None, "ci95_low": None, "ci95_high": None}

    arr = np.asarray(clean, dtype=np.float64)
    n = arr.size
    mean = float(arr.mean())
    sd = float(arr.std(ddof=1)) if n > 1 else 0.0

    if n > 1:
        half = _t_critical(n - 1) * sd / float(np.sqrt(n))
        low, high = mean - half, mean + half
    else:
        low = high = None

    return {
        "mean": mean,
        "median": float(np.median(arr)),
        "sd": sd,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "ci95_low": low,
        "ci95_high": high,
    }


def latency_stats(latencies_ms: Sequence[float]) -> dict[str, float]:
    """Average / median / percentile latency summary."""
    if not latencies_ms:
        return {
            "avg": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0,
        }
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "avg": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }
