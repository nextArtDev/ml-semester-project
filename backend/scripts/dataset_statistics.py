"""Emit dataset-level statistics used to interpret the benchmark.

Measures, rather than assumes:

* a stable fingerprint of the training pool an experiment ran against;
* duplicate feature-vector counts (groups, largest group, label conflicts);
* how many queries have a tied exact K-th neighbour distance, and whether a
  tie-only candidate would change the majority vote.

Usage::

    python -m scripts.dataset_statistics --dataset-id 116 \
        --k 5 --metric l2 --max-queries 1000 \
        --output ../docs/results/duplicate_stats.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.services.dataset_stats import (  # noqa: E402
    dataset_fingerprint,
    duplicate_vector_stats,
    tie_statistics,
)
from app.services.metrics import DistanceMetric  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure dataset statistics.")
    parser.add_argument("--dataset-id", type=int, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--metric", choices=["l2", "cosine", "l1"], default="l2")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1000,
        help="How many held-out queries to score for tied boundaries.",
    )
    parser.add_argument("--output", default="", help="JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metric = DistanceMetric(args.metric)

    payload = {
        "dataset_id": args.dataset_id,
        "dataset_fingerprint": dataset_fingerprint(args.dataset_id),
        "duplicates": duplicate_vector_stats(args.dataset_id),
        "ties": tie_statistics(
            args.dataset_id,
            k=args.k,
            metric=metric,
            limit=args.max_queries,
        ),
    }

    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
        print(f"JSON written to {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
