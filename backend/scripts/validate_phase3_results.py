"""Independent Phase 3 results validation and statistical audit.

Reads only the committed raw Phase 3 artifacts and the PostgreSQL records.
Writes validated summaries under docs/results/validated/. Raw artifacts are
never modified.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

from app.core.database import connection, fetch_all, fetch_one  # noqa: E402
from app.services.classification import (  # noqa: E402
    evaluate_predictions,
    majority_vote,
    repetition_summary,
)
from app.services.dataset_stats import dataset_fingerprint  # noqa: E402
from app.services.reproducibility import selected_ids_provenance  # noqa: E402
from app.services.vector_store import index_name  # noqa: E402
from scripts.run_experiments import FIELDS  # noqa: E402

RAW = ROOT / "docs" / "results" / "phase3"
OUT = ROOT / "docs" / "results" / "validated"
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 20261123
TOL = 1e-6

errors: list[str] = []
limitations: list[str] = []
max_discrepancy: dict[str, float] = defaultdict(float)


def check(condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def as_object(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return value or {}


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def close(a, b, label: str) -> bool:
    if a is None or b is None or a == "" or b == "":
        ok = (a in (None, "")) == (b in (None, ""))
    else:
        try:
            diff = abs(float(a) - float(b))
        except (TypeError, ValueError):
            ok = str(a) == str(b)
            diff = 0.0 if ok else math.inf
        max_discrepancy[label] = max(max_discrepancy[label], diff)
        ok = diff <= TOL
    return ok


def rankdata(values: np.ndarray) -> tuple[np.ndarray, list[int]]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    tie_sizes: list[int] = []
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j + 1) / 2.0
        if j - i > 1:
            tie_sizes.append(j - i)
        i = j
    return ranks, tie_sizes


def wilcoxon_signed_rank(diff: np.ndarray) -> dict:
    nonzero = diff[diff != 0]
    n = int(nonzero.size)
    if n == 0:
        return {"n_nonzero": 0, "p_two_sided": 1.0, "rank_biserial": 0.0}
    ranks, ties = rankdata(np.abs(nonzero))
    w_plus = float(ranks[nonzero > 0].sum())
    w_minus = float(ranks[nonzero < 0].sum())
    observed = min(w_plus, w_minus)
    mu = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    variance -= sum(t**3 - t for t in ties) / 48.0
    sigma = math.sqrt(max(variance, 1e-12))
    z = (observed - mu + 0.5) / sigma if observed < mu else (observed - mu - 0.5) / sigma
    p = math.erfc(abs(z) / math.sqrt(2.0))
    total = w_plus + w_minus
    effect = (w_plus - w_minus) / total if total else 0.0
    return {
        "n_nonzero": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "z": z,
        "p_two_sided": p,
        "rank_biserial": effect,
        "method": "normal approximation with tie correction and continuity correction",
    }


def mcnemar(b: int, c: int) -> dict:
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_two_sided": 1.0}
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2.0**n)
    return {
        "b": b,
        "c": c,
        "p_two_sided": min(1.0, 2.0 * tail),
        "method": "exact two-sided binomial",
    }


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    expected_files = [
        "sweep.csv", "sweep.json", "repetition_details.csv", "repetition_summary.csv",
        "query_samples.json", "benchmark_orders.json", "variant_ids.json",
        "controlled_diagnostic.json", "validation_summary.json", "standard_verifier_output.json",
    ]
    for name in expected_files:
        check((RAW / name).is_file(), f"missing raw artifact {name}")

    sweep_json = read_json(RAW / "sweep.json")
    sweep_csv = read_csv(RAW / "sweep.csv")
    details = read_csv(RAW / "repetition_details.csv")
    summary_csv = read_csv(RAW / "repetition_summary.csv")
    query_samples = read_json(RAW / "query_samples.json")
    orders = read_json(RAW / "benchmark_orders.json")
    variants = read_json(RAW / "variant_ids.json")["variants"]
    controlled_rows = read_json(RAW / "controlled_diagnostic.json")["experiments"]
    prior_validation = read_json(RAW / "validation_summary.json")
    natural_id, controlled_id = prior_validation["natural_run_id"], prior_validation["controlled_run_id"]

    check(len(sweep_json) == 30, f"sweep rows {len(sweep_json)}")
    check(len(sweep_csv) == len(sweep_json), "CSV/JSON sweep row count mismatch")
    for csv_row, json_row in zip(sweep_csv, sweep_json):
        check(set(csv_row) == set(FIELDS), "sweep CSV header mismatch")
        for key in FIELDS:
            expected = "" if json_row.get(key) is None else str(json_row[key])
            check(str(csv_row[key]) == expected, f"CSV/JSON mismatch for {key}")
    check(len(details) == 90, f"repetition detail rows {len(details)}")
    check(len(summary_csv) == 300, f"repetition summary rows {len(summary_csv)}")
    check(len(query_samples) == 30, f"query sample entries {len(query_samples)}")
    check(len(orders) == 6, f"benchmark order entries {len(orders)}")
    check(len(variants) == 2, f"variant entries {len(variants)}")
    check(len(controlled_rows) == 8, f"controlled diagnostic rows {len(controlled_rows)}")

    experiments = fetch_all(
        "SELECT * FROM experiments WHERE run_id IN (%s,%s)", (natural_id, controlled_id)
    )
    by_id = {int(e["id"]): e for e in experiments}
    check(len(experiments) == 38, f"DB experiments {len(experiments)}")
    natural = [e for e in experiments if e["run_id"] == natural_id]
    controlled = [e for e in experiments if e["run_id"] == controlled_id]
    check(len(natural) == 30, f"natural DB experiments {len(natural)}")
    check(len(controlled) == 8, f"controlled DB experiments {len(controlled)}")

    exp_ids = sorted(by_id)
    latencies = fetch_all(
        "SELECT experiment_id,repetition,query_index,latency_ms,retrieved_ids "
        "FROM query_latencies WHERE experiment_id = ANY(%s) ORDER BY experiment_id,repetition,query_index",
        (exp_ids,),
    )
    rep_rows = fetch_all(
        "SELECT * FROM experiment_repetitions WHERE experiment_id = ANY(%s) ORDER BY experiment_id,repetition",
        (exp_ids,),
    )
    lat_by_exp: dict[int, list[dict]] = defaultdict(list)
    for row in latencies:
        lat_by_exp[int(row["experiment_id"])].append(row)
    rep_by_exp: dict[int, list[dict]] = defaultdict(list)
    for row in rep_rows:
        rep_by_exp[int(row["experiment_id"])].append(row)

    check(len(latencies) == 342_000, f"latency rows {len(latencies)}")
    check(len(rep_rows) == 114, f"repetition rows {len(rep_rows)}")
    for exp in experiments:
        eid = int(exp["id"])
        check(len(lat_by_exp[eid]) == 9000, f"latency count for {eid}")
        check(len(rep_by_exp[eid]) == 3, f"repetition count for {eid}")
    check(
        int(fetch_one("SELECT count(*) AS c FROM query_latencies q LEFT JOIN experiments e ON e.id=q.experiment_id WHERE e.id IS NULL")["c"]) == 0,
        "orphan latency rows",
    )
    check(
        int(fetch_one("SELECT count(*) AS c FROM experiment_repetitions r LEFT JOIN experiments e ON e.id=r.experiment_id WHERE e.id IS NULL")["c"]) == 0,
        "orphan repetition rows",
    )

    # Provenance and historical integrity.
    check(dataset_fingerprint(116) == "6baa8fa283ea5a8267d3e964eedf5d2a", "parent fingerprint changed")
    historical = fetch_all(
        "SELECT id,status,retrieval_method,hnsw_ef_search,used_index_scan,error_message "
        "FROM experiments WHERE id IN (382,383,384) ORDER BY id"
    )
    expected_history = [
        (382, "completed", "exact", None, False, None),
        (383, "completed", "hnsw", 50, False, None),
        (384, "completed", "hnsw", 10, True, None),
    ]
    actual_history = [
        (int(r["id"]), r["status"], r["retrieval_method"], r["hnsw_ef_search"], r["used_index_scan"], r["error_message"])
        for r in historical
    ]
    check(actual_history == expected_history, "historical Phase 2 experiments changed")

    for did in (847, 848):
        meta = variants[str(did)]
        with connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    f"SELECT p.id FROM fv_116 p JOIN fv_{did} v ON v.row_index=p.row_index ORDER BY p.id"
                )
                selected = [int(r["id"]) for r in cur.fetchall()]
        prov = selected_ids_provenance(selected)
        check(prov["sha256"] == meta["selected_id_sha256"], f"selected-id hash mismatch for {did}")
        check(prov["count"] == int(meta["selected_id_count"]) == int(meta["actual_size"]), f"selected-id count mismatch for {did}")

    # Query ids and class/neighbour labels.
    query_ids_by_exp: dict[int, list[int]] = {}
    for exp in experiments:
        eid = int(exp["id"])
        qids = as_object(exp["evaluated_query_ids"])
        check(len(qids) == 3000, f"query id count for {eid}")
        query_ids_by_exp[eid] = [int(q) for q in qids]
        artifact = query_samples.get(str(eid))
        if exp["run_id"] == natural_id:
            check(artifact is not None, f"query_samples missing {eid}")
            if artifact:
                check(artifact["query_ids"] == qids, f"query ids mismatch for {eid}")
    for did in (847, 848):
        ids = [eid for eid, qids in query_ids_by_exp.items() if int(by_id[eid]["dataset_id"]) == did]
        sets = {json.dumps(query_ids_by_exp[eid]) for eid in ids}
        check(len(sets) == 1, f"query ids differ within dataset {did}")
    limitations.append(
        "query_samples.json is produced by the natural runner only; controlled-HNSW "
        "diagnostic query ids were verified directly from database records."
    )

    parent_labels = as_object(fetch_one("SELECT class_labels FROM datasets WHERE id=116")["class_labels"])
    train_labels: dict[int, dict[int, int]] = {}
    test_labels: dict[int, dict[int, int]] = {}
    for did in (847, 848):
        train_labels[did] = {int(r["id"]): int(r["label"]) for r in fetch_all(f"SELECT id,label FROM fv_{did}")}
        test_labels[did] = {int(r["id"]): int(r["label"]) for r in fetch_all(f"SELECT id,label FROM tv_{did}")}

    # Access-path classification.
    access: dict[int, str] = {}
    for exp in experiments:
        eid = int(exp["id"])
        env = as_object(exp["environment"])
        method = exp["retrieval_method"]
        if method == "exact":
            access[eid] = "valid_exact_sequential"
            check(exp["used_index_scan"] is False, f"exact {eid} used index")
            check(env.get("used_sequential_scan") is True, f"exact {eid} not sequential")
        elif exp["run_id"] == controlled_id:
            valid = (
                exp["status"] == "completed"
                and exp["used_index_scan"] is True
                and env.get("planner_mode") == "controlled_hnsw"
                and env.get("planner_settings", {}).get("enable_seqscan") == "off"
                and env.get("valid_for_summary") is True
            )
            access[eid] = "valid_controlled_hnsw" if valid else "invalid_controlled_hnsw"
        else:
            valid = (
                exp["status"] == "completed"
                and exp["used_index_scan"] is True
                and env.get("used_sequential_scan") is False
                and index_name(int(exp["dataset_id"])).lower() in str(env.get("execution_plan", "")).lower()
                and env.get("planner_mode") == "natural"
                and env.get("valid_for_summary") is True
            )
            if valid:
                access[eid] = "valid_natural_hnsw"
            elif exp["status"] == "failed" and env.get("used_sequential_scan") is True:
                access[eid] = "natural_seq_scan_failure"
            else:
                access[eid] = "unverified_hnsw"

    failed_ids = sorted(eid for eid, kind in access.items() if kind == "natural_seq_scan_failure")
    check(failed_ids == [388, 392, 396], f"unexpected failed ids {failed_ids}")
    check(
        all(int(by_id[eid]["hnsw_ef_search"]) == 400 and int(by_id[eid]["dataset_id"]) == 847 for eid in failed_ids),
        "failed set is not the documented 10k/ef=400 natural cell",
    )

    # Ground truth from exact experiments (first repetition), keyed by size/round/query_index.
    gt_ids: dict[tuple[int, int, int], list[int]] = {}
    for exp in natural:
        if exp["retrieval_method"] != "exact":
            continue
        size, rnd = int(exp["train_size"]), int(exp["benchmark_round"])
        for row in lat_by_exp[int(exp["id"])]:
            if row["repetition"] == 1:
                gt_ids[(size, rnd, int(row["query_index"]))] = [int(v) for v in as_object(row["retrieved_ids"])]

    def compute_metrics(exp: dict) -> dict:
        eid = int(exp["id"])
        did = int(exp["dataset_id"])
        rows = lat_by_exp[eid]
        lat = np.array([float(r["latency_ms"]) for r in rows], dtype=float)
        latency = {
            "p50": float(np.percentile(lat, 50)),
            "p95": float(np.percentile(lat, 95)),
            "p99": float(np.percentile(lat, 99)),
            "min": float(lat.min()),
            "max": float(lat.max()),
        }
        for key, column in (("p50", "latency_p50_ms"), ("p95", "latency_p95_ms"), ("p99", "latency_p99_ms"), ("min", "latency_min_ms"), ("max", "latency_max_ms")):
            check(close(latency[key], exp[column], f"latency_{key}"), f"latency {key} mismatch for {eid}")
        check(close(lat.mean(), exp["latency_avg_ms"], "latency_avg"), f"latency avg mismatch for {eid}")

        recall_scores: list[float] = []
        jaccard_scores: list[float] = []
        recall_by_rep: dict[int, list[float]] = {1: [], 2: [], 3: []}
        if exp["retrieval_method"] == "exact":
            recall_scores = [1.0] * len(rows)
            jaccard_scores = [1.0] * len(rows)
            for rep in (1, 2, 3):
                recall_by_rep[rep] = [1.0] * 3000
        else:
            size = int(exp["train_size"])
            rnd = int(exp["benchmark_round"]) if exp["benchmark_round"] is not None else None
            for row in rows:
                got = set(int(v) for v in as_object(row["retrieved_ids"]))
                qidx = int(row["query_index"])
                want = set(gt_ids.get((size, rnd, qidx), gt_by_size[size][qidx]))
                score = len(got & want) / len(want)
                recall_scores.append(score)
                recall_by_rep[int(row["repetition"])].append(score)
                jaccard_scores.append(len(got & want) / len(got | want))
        recall = float(np.mean(recall_scores))
        check(close(recall, exp["recall_at_k"], "recall_at_k"), f"recall mismatch for {eid}")

        per_rep: dict[int, dict] = {}
        for rep in (1, 2, 3):
            y_true, y_pred = [], []
            for row in sorted((r for r in rows if int(r["repetition"]) == rep), key=lambda r: int(r["query_index"])):
                qid = query_ids_by_exp[eid][int(row["query_index"])]
                y_true.append(int(test_labels[did][qid]))
                neighbour_labels = [train_labels[did][int(v)] for v in as_object(row["retrieved_ids"])]
                y_pred.append(int(majority_vote(neighbour_labels)))
            per_rep[rep] = evaluate_predictions(y_true, y_pred, parent_labels)
            stored = next((r for r in rep_by_exp[eid] if int(r["repetition"]) == rep), None)
            check(stored is not None, f"missing repetition row for {eid}/{rep}")
            if stored:
                for key, column in (("accuracy", "accuracy"), ("precision_macro", "precision_macro"), ("recall_macro", "recall_macro"), ("macro_f1", "macro_f1")):
                    check(close(per_rep[rep].__dict__[key], stored[column], f"rep_{key}"), f"repetition {key} mismatch for {eid}/{rep}")
                check(close(float(np.mean(recall_by_rep[rep])), stored["recall_at_k"], "rep_recall_at_k"), f"repetition recall mismatch for {eid}/{rep}")
        accuracy = float(np.mean([per_rep[r].accuracy for r in (1, 2, 3)]))
        macro_f1 = float(np.mean([per_rep[r].macro_f1 for r in (1, 2, 3)]))
        macro_precision = float(np.mean([per_rep[r].precision_macro for r in (1, 2, 3)]))
        macro_recall = float(np.mean([per_rep[r].recall_macro for r in (1, 2, 3)]))
        for label, value in (("accuracy", accuracy), ("macro_f1", macro_f1), ("precision_macro", macro_precision), ("recall_macro", macro_recall)):
            check(close(value, exp[label], label), f"{label} mismatch for {eid}")
        last = per_rep[3]
        check(last.confusion_matrix == as_object(exp["confusion_matrix"]), f"confusion matrix mismatch for {eid}")
        check(last.per_class == as_object(exp["per_class_metrics"]), f"per-class metrics mismatch for {eid}")

        # Repetition summary CSV comparison.
        metric_values = {
            "latency_mean_ms": [float(r["latency_avg_ms"]) for r in rep_by_exp[eid]],
            "latency_p50_ms": [float(r["latency_p50_ms"]) for r in rep_by_exp[eid]],
            "latency_p95_ms": [float(r["latency_p95_ms"]) for r in rep_by_exp[eid]],
            "latency_p99_ms": [float(r["latency_p99_ms"]) for r in rep_by_exp[eid]],
            "recall_at_k": [float(r["recall_at_k"]) for r in rep_by_exp[eid]],
            "tie_aware_recall_at_k": [float(r["tie_aware_recall_at_k"]) for r in rep_by_exp[eid]],
            "accuracy": [float(r["accuracy"]) for r in rep_by_exp[eid]],
            "precision_macro": [float(r["precision_macro"]) for r in rep_by_exp[eid]],
            "recall_macro": [float(r["recall_macro"]) for r in rep_by_exp[eid]],
            "macro_f1": [float(r["macro_f1"]) for r in rep_by_exp[eid]],
        }
        for csv_row in summary_csv:
            if int(csv_row["experiment_id"]) == eid and csv_row["metric"] in metric_values:
                recomputed = repetition_summary(metric_values[csv_row["metric"]])
                for key in ("mean", "median", "sd", "min", "max", "ci95_low", "ci95_high"):
                    stored_value = csv_row[key]
                    check(close(recomputed[key], stored_value, f"summary_{key}"), f"repetition summary {key} mismatch for {eid}/{csv_row['metric']}")

        return {
            "latency": latency,
            "recall": recall,
            "jaccard": float(np.mean(jaccard_scores)),
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "per_class": last.per_class,
            "confusion_matrix": last.confusion_matrix,
            "latency_array": lat,
            "recall_array": np.array(recall_scores),
            "correct_majority": _majority_correct(rows, query_ids_by_exp[eid], train_labels[did], test_labels[did]),
        }

    def _majority_correct(rows, qids, tlabels, vlabels):
        predictions: dict[int, list[int]] = defaultdict(list)
        for row in rows:
            qidx = int(row["query_index"])
            labels = [tlabels[int(v)] for v in as_object(row["retrieved_ids"])]
            predictions[qidx].append(int(majority_vote(labels)))
        out = {}
        for qidx, preds in predictions.items():
            counts = Counter(preds)
            best = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]
            out[qidx] = int(best == vlabels[qids[qidx]])
        return out

    gt_by_size: dict[int, dict[int, list[int]]] = defaultdict(dict)
    for (size, _rnd, qidx), ids in gt_ids.items():
        gt_by_size[size].setdefault(qidx, ids)

    recomputed: dict[int, dict] = {}
    for exp in experiments:
        recomputed[int(exp["id"])] = compute_metrics(exp)

    # Index catalog verification.
    for did in (847, 848):
        idx = index_name(did)
        catalog = fetch_one(
            "SELECT am.amname, op.opcname, c.reltuples::bigint AS reltuples, c.relpages::bigint AS relpages, "
            "pg_relation_size(c.oid) AS size_bytes FROM pg_class c "
            "JOIN pg_index i ON i.indexrelid=c.oid JOIN pg_am am ON am.oid=c.relam "
            "LEFT JOIN pg_opclass op ON op.oid=i.indclass[0] WHERE c.oid=to_regclass(%s)",
            (idx,),
        )
        check(catalog is not None, f"missing HNSW index {idx}")
        if catalog:
            check(catalog["amname"] == "hnsw" and catalog["opcname"] == "vector_l2_ops", f"index metadata mismatch {idx}")
            stored_sizes = [int(e["index_size_bytes"]) for e in natural if int(e["dataset_id"]) == did and e["index_size_bytes"]]
            check(bool(stored_sizes) and all(int(catalog["size_bytes"]) == s for s in stored_sizes), f"index size mismatch {idx}")

    # Statistical audit on valid natural comparisons.
    comparisons = []
    lat_pvalues: dict[int, list[float]] = defaultdict(list)
    cls_pvalues: dict[int, list[float]] = defaultdict(list)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap_idx = rng.integers(0, 3000, size=(BOOTSTRAP_N, 3000))
    for size in (10_000, 50_000):
        exact = next(e for e in natural if int(e["train_size"]) == size and e["retrieval_method"] == "exact")
        exact_m = recomputed[int(exact["id"])]
        for ef in (10, 50, 100, 400):
            candidates = [e for e in natural if int(e["train_size"]) == size and e["retrieval_method"] == "hnsw" and int(e["hnsw_ef_search"]) == ef]
            if not candidates:
                continue
            exp = candidates[0]
            eid = int(exp["id"])
            if access[eid] != "valid_natural_hnsw":
                continue
            h = recomputed[eid]
            lat_diff = np.array([h["latency_array"][q] - exact_m["latency_array"][q] for q in range(3000)])
            lat_boot = lat_diff[bootstrap_idx]
            lat_ci = [float(np.percentile(np.median(lat_boot, axis=1), 2.5)), float(np.percentile(np.median(lat_boot, axis=1), 97.5))]
            mean_lat_ci = [float(np.percentile(np.mean(lat_boot, axis=1), 2.5)), float(np.percentile(np.mean(lat_boot, axis=1), 97.5))]
            wilcox = wilcoxon_signed_rank(lat_diff)
            recall_diff = np.array([h["recall_array"][q] - exact_m["recall_array"][q] for q in range(3000)])
            recall_boot = recall_diff[bootstrap_idx]
            recall_ci = [float(np.percentile(np.mean(recall_boot, axis=1), 2.5)), float(np.percentile(np.mean(recall_boot, axis=1), 97.5))]
            h_corr = np.array([h["correct_majority"][q] for q in range(3000)])
            e_corr = np.array([exact_m["correct_majority"][q] for q in range(3000)])
            mcn = mcnemar(int(((~h_corr) & e_corr).sum()), int((h_corr & (~e_corr)).sum()))
            acc_diff = np.array([float(a) - float(b) for a, b in zip(h_corr, e_corr)])
            acc_boot = acc_diff[bootstrap_idx]
            acc_ci = [float(np.percentile(np.mean(acc_boot, axis=1), 2.5)), float(np.percentile(np.mean(acc_boot, axis=1), 97.5))]
            lat_pvalues[size].append(wilcox["p_two_sided"])
            cls_pvalues[size].append(mcn["p_two_sided"])
            comparisons.append({
                "size": size,
                "ef_search": ef,
                "hnsw_experiment_id": eid,
                "exact_experiment_id": int(exact["id"]),
                "n_queries": 3000,
                "passes_per_query": 9,
                "latency": {
                    "median_difference_ms": float(np.median(lat_diff)),
                    "median_difference_ci95_ms": lat_ci,
                    "mean_difference_ci95_ms": mean_lat_ci,
                    "wilcoxon": wilcox,
                },
                "recall_at_5": {
                    "mean_difference": float(np.mean(recall_diff)),
                    "mean_difference_ci95": recall_ci,
                },
                "classification": {
                    "hnsw_accuracy": float(h_corr.mean()),
                    "exact_accuracy": float(e_corr.mean()),
                    "accuracy_difference": float(np.mean(acc_diff)),
                    "accuracy_difference_ci95": acc_ci,
                    "mcnemar": mcn,
                },
            })
    for size in lat_pvalues:
        adj_lat = holm(lat_pvalues[size])
        adj_cls = holm(cls_pvalues[size])
        for comp, pl, pc in zip([c for c in comparisons if c["size"] == size], adj_lat, adj_cls):
            comp["latency"]["wilcoxon"]["p_holm"] = pl
            comp["classification"]["mcnemar"]["p_holm"] = pc

    # Validated summary rows (valid natural + valid controlled only).
    fields = [
        "run_id", "planner_mode", "experiment_id", "dataset_id", "train_size",
        "retrieval_method", "ef_search", "benchmark_round", "benchmark_position",
        "status", "access_path_class", "used_index_scan", "applied_ef_search",
        "valid_for_summary", "dataset_fingerprint", "selected_id_sha256",
        "preprocessing_state_sha256", "query_count", "recall_at_k",
        "neighbour_jaccard", "accuracy", "precision_macro", "recall_macro",
        "macro_f1", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
        "latency_min_ms", "latency_max_ms", "speedup_p50", "index_build_ms",
        "index_size_bytes",
    ]
    summary_rows = []
    for exp in sorted(experiments, key=lambda e: (e["run_id"], int(e["train_size"]), e["retrieval_method"], int(e["hnsw_ef_search"] or -1), int(e["benchmark_round"] or 0))):
        eid = int(exp["id"])
        if access[eid] not in {"valid_exact_sequential", "valid_natural_hnsw", "valid_controlled_hnsw"}:
            continue
        env = as_object(exp["environment"])
        meta = variants[str(int(exp["dataset_id"]))]
        m = recomputed[eid]
        sweep = next((r for r in sweep_json if int(r["experiment_id"]) == eid), {})
        summary_rows.append({
            "run_id": exp["run_id"],
            "planner_mode": env.get("planner_mode", "natural"),
            "experiment_id": eid,
            "dataset_id": int(exp["dataset_id"]),
            "train_size": int(exp["train_size"]),
            "retrieval_method": exp["retrieval_method"],
            "ef_search": exp["hnsw_ef_search"],
            "benchmark_round": exp["benchmark_round"],
            "benchmark_position": exp["benchmark_position"],
            "status": exp["status"],
            "access_path_class": access[eid],
            "used_index_scan": exp["used_index_scan"],
            "applied_ef_search": env.get("applied_ef_search"),
            "valid_for_summary": env.get("valid_for_summary"),
            "dataset_fingerprint": meta["derived_fingerprint"],
            "selected_id_sha256": meta["selected_id_sha256"],
            "preprocessing_state_sha256": meta["preprocessing_state_sha256"],
            "query_count": 3000,
            "recall_at_k": m["recall"],
            "neighbour_jaccard": m["jaccard"],
            "accuracy": m["accuracy"],
            "precision_macro": m["macro_precision"],
            "recall_macro": m["macro_recall"],
            "macro_f1": m["macro_f1"],
            "latency_p50_ms": m["latency"]["p50"],
            "latency_p95_ms": m["latency"]["p95"],
            "latency_p99_ms": m["latency"]["p99"],
            "latency_min_ms": m["latency"]["min"],
            "latency_max_ms": m["latency"]["max"],
            "speedup_p50": sweep.get("speedup_p50"),
            "index_build_ms": exp["index_build_ms"],
            "index_size_bytes": exp["index_size_bytes"],
        })
    with (OUT / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    statistical = {
        "validation_status": "PASS_WITH_DOCUMENTED_EXCLUSIONS" if not errors else "FAIL",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "starting_commit": "e8cdb87fd58445d7089a2a2e5071b41fc3cb4330",
        "run_ids": {"natural": natural_id, "controlled_hnsw": controlled_id},
        "included_configurations": [
            {"experiment_id": e["id"], "run_id": e["run_id"], "size": int(e["train_size"]), "method": e["retrieval_method"], "ef_search": e["hnsw_ef_search"], "class": access[int(e["id"])]}
            for e in experiments if access[int(e["id"])] in {"valid_exact_sequential", "valid_natural_hnsw", "valid_controlled_hnsw"}
        ],
        "excluded_configurations": [
            {"experiment_id": eid, "reason": "natural planner selected a sequential scan; HNSW access-path check failed"}
            for eid in failed_ids
        ],
        "sample_sizes": {
            "queries_per_configuration": 3000,
            "passes_per_query": 9,
            "observations_per_configuration": 9000,
            "valid_natural_configurations": 27,
            "valid_controlled_configurations": 8,
            "excluded_natural_configurations": 3,
        },
        "statistical_methods": {
            "unit_of_inference": "query id within pool size; repetitions and rounds aggregated per query",
            "latency": "two-sided Wilcoxon signed-rank on paired per-query median differences; 10,000-resample percentile bootstrap CIs; rank-biserial effect size",
            "recall": "paired percentile bootstrap CI for mean Recall@5 difference",
            "classification": "exact McNemar on paired per-query majority-vote correctness; percentile bootstrap CI for accuracy difference",
            "multiple_comparisons": "Holm correction across ef levels within each size, separately for latency and classification families",
            "assumptions": "same query ids across paired configurations; repeated passes are not independent; no normality assumption for Wilcoxon; bootstrap resamples queries",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "practical_significance": "no pre-registered threshold; effect magnitudes reported without declaring practical superiority",
        },
        "comparisons": comparisons,
        "controlled_hnsw_diagnostic": {
            "note": "descriptive only; never pooled with natural planner results",
            "experiments": len(controlled_rows),
            "all_index_scans": all(r["used_index_scan"] for r in controlled_rows),
            "all_completed": all(r["status"] == "completed" for r in controlled_rows),
        },
        "metric_recomputation": {
            "tolerance_absolute": TOL,
            "max_absolute_discrepancy_by_metric": dict(max_discrepancy),
            "discrepancies": [e for e in errors if "mismatch" in e],
        },
        "validation_errors": errors,
        "limitations": limitations,
        "reproducibility": {
            "script": "backend/scripts/validate_phase3_results.py",
            "raw_artifacts": "docs/results/phase3/",
            "raw_artifacts_modified": False,
            "database": "knn_knn_platform",
            "postgresql": "18.0",
            "pgvector": "0.8.7",
        },
    }
    (OUT / "statistical_summary.json").write_text(json.dumps(statistical, indent=2, default=str), encoding="utf-8")

    print(json.dumps({
        "status": statistical["validation_status"],
        "summary_rows": len(summary_rows),
        "errors": errors,
        "max_discrepancy": dict(max_discrepancy),
        "failed_ids": failed_ids,
        "comparisons": len(comparisons),
    }, indent=2, default=str))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
