"""Validate that the published result artifacts match the stored experiments.

This script is deliberately strict: it checks the experiment count, balanced
rounds, per-configuration coverage, index metadata, repetition and latency row
counts, query ids and class distributions.  It reads numbers only from the
generated artifacts and the database; it never edits them.

Usage::

    python -m scripts.verify_results --run-id final-v3 --expected-runs 42 ...
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.database import fetch_all, fetch_one  # noqa: E402
from scripts.run_experiments import FIELDS, REPETITION_FIELDS, SUMMARY_FIELDS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify benchmark artifacts.")
    parser.add_argument("--results-dir", default="../docs/results")
    parser.add_argument("--details-name", default="repetition_details.csv")
    parser.add_argument("--summary-name", default="repetition_summary.csv")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-runs", type=int, default=42)
    parser.add_argument("--expected-rounds", type=int, default=3)
    parser.add_argument("--expected-queries", type=int, default=3000)
    parser.add_argument("--expected-repetitions", type=int, default=3)
    return parser.parse_args()


def _csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _as_object(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def main() -> int:
    args = parse_args()
    directory = Path(args.results_dir)
    sweep_path = directory / "sweep.csv"
    json_path = directory / "sweep.json"
    details_path = directory / args.details_name
    summary_path = directory / args.summary_name
    samples_path = directory / "query_samples.json"
    orders_path = directory / "benchmark_orders.json"

    rows = _csv(sweep_path)
    json_rows = json.loads(json_path.read_text(encoding="utf-8"))
    details = _csv(details_path)
    summary = _csv(summary_path)
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    orders = json.loads(orders_path.read_text(encoding="utf-8"))

    errors: list[str] = []

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    # --- artifact shape ---------------------------------------------------
    with sweep_path.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh))
    check(header == FIELDS, "sweep.csv header does not match runner.FIELDS")
    check("api_overhead_s" not in header, "obsolete api_overhead_s column present")
    for path, expected in (
        (details_path, REPETITION_FIELDS),
        (summary_path, SUMMARY_FIELDS),
    ):
        with path.open(newline="", encoding="utf-8") as fh:
            check(next(csv.reader(fh)) == expected, f"{path.name} header mismatch")

    check(len(rows) == args.expected_runs, f"expected {args.expected_runs} runs, got {len(rows)}")
    check(len(json_rows) == len(rows), "JSON row count differs from CSV")
    check([str(r["experiment_id"]) for r in rows] == [str(r["experiment_id"]) for r in json_rows],
          "CSV and JSON experiment ids differ")

    ids = [int(r["experiment_id"]) for r in rows]
    check(len(set(ids)) == len(ids), "duplicate experiment ids in sweep.csv")
    placeholders = ",".join(["%s"] * len(ids))
    db_rows = fetch_all(
        f"SELECT * FROM experiments WHERE id IN ({placeholders}) ORDER BY id", tuple(ids)
    )
    check(len(db_rows) == len(ids), "not every CSV experiment id exists in the database")
    db_by_id = {int(r["id"]): r for r in db_rows}

    for row, json_row in zip(rows, json_rows):
        for key in FIELDS:
            expected = "" if json_row[key] is None else str(json_row[key])
            check(str(row[key]) == expected, f"CSV/JSON mismatch: {key} for {row['experiment_id']}")
        record = db_by_id.get(int(row["experiment_id"]))
        if record is None:
            continue
        check(record["run_id"] == args.run_id, f"wrong run_id for {row['experiment_id']}")
        check(record["dataset_fingerprint"] == row["dataset_fingerprint"],
              f"fingerprint mismatch for {row['experiment_id']}")
        check(int(record["unique_queries"]) == args.expected_queries,
              f"query count mismatch for {row['experiment_id']}")
        check(int(record["repetitions"]) == args.expected_repetitions,
              f"repetition count mismatch for {row['experiment_id']}")

        environment = _as_object(record["environment"]) or {}
        if row["retrieval_method"] == "hnsw":
            check(record["used_index_scan"] is True,
                  f"HNSW row {row['experiment_id']} did not use the index")
            check(row["hnsw_m"] not in ("", "None"), f"missing hnsw_m for {row['experiment_id']}")
            check(row["hnsw_ef_construction"] not in ("", "None"),
                  f"missing hnsw_ef_construction for {row['experiment_id']}")
            check(int(environment["applied_ef_search"]) == int(row["hnsw_ef_search"]),
                  f"applied ef mismatch for {row['experiment_id']}")
            check(row["index_size_bytes"] not in ("", "None"),
                  f"missing index size for {row['experiment_id']}")
        else:
            check(record["used_index_scan"] is False,
                  f"exact row {row['experiment_id']} used an index")
            check(environment.get("used_sequential_scan") is True,
                  f"exact row {row['experiment_id']} was not a sequential scan")
            check(row["hnsw_ef_search"] in ("", "None"),
                  f"exact row {row['experiment_id']} has an ef_search")
            check(row["index_build_ms"] in ("", "None"),
                  f"exact row {row['experiment_id']} reports an index build")

        counts = fetch_one(
            "SELECT count(*) AS latencies, count(DISTINCT repetition) AS reps "
            "FROM query_latencies WHERE experiment_id = %s", (record["id"],)
        )
        expected_latencies = args.expected_repetitions * args.expected_queries
        check(int(counts["latencies"]) == expected_latencies,
              f"wrong latency count for {record['id']}")
        check(int(counts["reps"]) == args.expected_repetitions,
              f"wrong repetition count for {record['id']}")

    # --- rounds and configurations ---------------------------------------
    expected_configs = {"exact:-"} | {
        f"hnsw:{ef}" for ef in (10, 25, 50, 100, 200, 400)
    }
    by_round: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        key = (row["train_size"], int(row["benchmark_round"]))
        by_round.setdefault(key, []).append(row)
    for (size, round_number), items in by_round.items():
        configs = {
            f"{r['retrieval_method']}:{r['hnsw_ef_search'] or '-'}" for r in items
        }
        check(configs == expected_configs,
              f"round {round_number} of size {size} is missing configurations")
        check(sorted(int(r["benchmark_position"]) for r in items) == list(range(1, len(items) + 1)),
              f"round {round_number} of size {size} has bad positions")
    check(len(by_round) == 2 * args.expected_rounds,
          f"expected {2 * args.expected_rounds} size/round groups, got {len(by_round)}")

    for key, payload in orders.items():
        check(payload["run_id"] == args.run_id, f"wrong run_id in benchmark_orders for {key}")
        check(sorted(payload["order"]) == sorted(expected_configs),
              f"benchmark_orders entry {key} is not a permutation")
        check(int(payload["train_size_actual"]) > 0, f"missing actual size for {key}")
        stored = fetch_all(
            "SELECT id FROM experiments WHERE id = ANY(%s) ORDER BY id",
            (payload["experiment_ids"],),
        )
        check([int(r["id"]) for r in stored] == payload["experiment_ids"],
              f"benchmark_orders ids not stored for {key}")

    # --- repetition/summary/query artifacts -------------------------------
    expected_detail_ids: list[int] = []
    for record in db_by_id.values():
        expected_detail_ids.extend([int(record["id"])] * args.expected_repetitions)
    check(sorted(int(r["experiment_id"]) for r in details) == sorted(expected_detail_ids),
          "repetition_details does not cover every experiment/repetition")
    check(len(summary) == args.expected_runs * 10, "repetition_summary row count is wrong")
    check(set(samples) == {str(i) for i in ids}, "query_samples keys differ from sweep ids")
    for experiment_id, payload in samples.items():
        record = db_by_id[int(experiment_id)]
        stored_ids = _as_object(record["evaluated_query_ids"])
        check(payload["query_ids"] == stored_ids,
              f"query ids differ from stored ids for {experiment_id}")
        check(len(payload["query_ids"]) == args.expected_queries,
              f"wrong exported query count for {experiment_id}")
        check(record["test_pool_class_distribution"] and record["evaluated_class_distribution"],
              f"missing class distribution for {experiment_id}")

    # --- cache and timing fields -----------------------------------------
    check(all(r["ground_truth_cache_hit"] in ("True", "False") for r in rows),
          "missing ground_truth_cache_hit values")
    check(all(r["retrieval_batch_wall_time_s"] not in ("", "None") for r in rows),
          "missing retrieval batch wall time")
    check(all(r["summed_query_execution_time_s"] == r["retrieval_total_s"] for r in rows),
          "summed query time differs from legacy retrieval_total_s")

    summary_out = {
        "run_id": args.run_id,
        "experiments": len(rows),
        "rounds": args.expected_rounds,
        "queries_per_experiment": args.expected_queries,
        "repetitions_per_experiment": args.expected_repetitions,
        "latency_observations": len(rows) * args.expected_queries * args.expected_repetitions,
        "errors": errors,
    }
    print(json.dumps(summary_out, indent=2))
    if errors:
        print(f"FAILED: {len(errors)} checks", file=sys.stderr)
        return 1
    print("VERIFIED: all artifact checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
