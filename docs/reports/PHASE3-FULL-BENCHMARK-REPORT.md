# Phase 3 — Full Benchmark Execution Report

**Branch:** `phase3-benchmark-design`  
**Protocol:** `docs/reports/PHASE3-BENCHMARK-PROTOCOL.md` @ `6895f24`  
**Execution approval:** Prompt 5 GO with option A  
**Natural run ID:** `phase3-natural-v1-20261123T224455`  
**Controlled run ID:** `phase3-controlled-diagnostic-v1-20261123T224455`  
**Full benchmark executed:** yes  
**Statistical analysis / thesis generation:** not performed

---

## 1. Preflight

| Check | Result |
|---|---|
| Branch / commit | `phase3-benchmark-design` @ `39c7125` before execution |
| Working tree | clean; stash `phase3-preflight-unrelated` present and untouched |
| Phase 2 GO | remediation GO; isolated diagnostic re-test completed at `39c7125` |
| PostgreSQL / pgvector | 18.0 / 0.8.7 |
| Resources | 39 GB free on C:; 31.9 GB total RAM, 17.2 GB available |
| Backup | `knn_knn_platform_phase3_pre_20261123T224455.dump`; SHA-256 `3f71c9d946245bb95b6a627b3857fe826be47fcccdfca38a293018d45f8c79b8`; `pg_restore --list` succeeded |
| Parent dataset | `dataset_fingerprint(116) = 6baa8fa283ea5a8267d3e964eedf5d2a` before and after |
| Historical records | Experiments 382/383/384 unchanged |

Created variants: 847 (10,000 rows) and 848 (50,000 rows), each with 3,000 held-out queries.

## 2. Execution outcome

| Category | Required | Completed | Valid HNSW path | Failed access-path check |
|---|---:|---:|---:|---:|
| Natural planner matrix | 30 | 30 | 24 of 26 HNSW runs | 3 |
| Controlled HNSW diagnostic | 8 | 8 | 8 of 8 | 0 |
| Total timed queries | 342,000 | 342,000 | — | — |

The standard runner was interrupted twice by the 10-minute command limit after writing experiment rows but before writing artifacts. **No completed experiment was deleted or overwritten.** Two resume passes executed only the missing 13 + 1 natural configurations, then rebuilt all artifacts from the preserved database rows. Raw measurements and failed runs were retained.

### Failed configurations (option A)

| Experiment | Size | ef | Round | Status | Access path | Error |
|---:|---:|---:|---:|---|---|---|
| 388 | 10,000 | 400 | 1 | `failed` | Seq Scan | `HNSW run did not use the verified HNSW index; HNSW run used a sequential scan` |
| 392 | 10,000 | 400 | 2 | `failed` | Seq Scan | same |
| 396 | 10,000 | 400 | 3 | `failed` | Seq Scan | same |

These are the expected natural-planner outcomes at 10k/ef=400. Their latencies are **not** HNSW measurements, have no speed-up, and are marked `valid_for_summary=false`. No index was forced for these natural runs. The separately identified controlled-HNSW ef=400 run at 10k did use the index and is not pooled with the natural result.

## 3. Metrics summary

Averages across the three rounds are shown. Failed natural rows are shown for completeness but excluded from valid HNSW interpretation.

| Size | Method | ef | Status | Recall@5 | Accuracy | Macro F1 | p50 ms | p95 ms | p99 ms | p50 speed-up | Index build ms | Index bytes |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | exact | — | completed | 1.0000 | 0.7547 | 0.5605 | 2.386 | 2.765 | 3.255 | — | — | — |
| 10,000 | hnsw | 10 | completed | 0.9949 | 0.7537 | 0.5588 | 0.298 | 0.398 | 0.483 | 8.01× | reused | 5,226,496 |
| 10,000 | hnsw | 50 | completed | 0.9998 | 0.7550 | 0.5606 | 0.431 | 0.579 | 0.700 | 5.56× | 1,478.5 | 5,226,496 |
| 10,000 | hnsw | 100 | completed | 0.9999 | 0.7547 | 0.5605 | 0.582 | 0.798 | 0.995 | 4.11× | reused | 5,226,496 |
| 10,000 | hnsw | 400 | **failed** | 0.9999 | 0.7547 | 0.5605 | 1.238 | 1.661 | 2.039 | — | reused | 5,226,496 |
| 50,000 | exact | — | completed | 1.0000 | 0.8440 | 0.7500 | 11.278 | 11.786 | 12.376 | — | — | — |
| 50,000 | hnsw | 10 | completed | 0.9829 | 0.8420 | 0.7460 | 0.285 | 0.361 | 0.424 | 39.63× | reused | 26,083,328 |
| 50,000 | hnsw | 50 | completed | 0.9975 | 0.8433 | 0.7498 | 0.416 | 0.539 | 0.617 | 27.17× | 3,500.2 | 26,083,328 |
| 50,000 | hnsw | 100 | completed | 0.9988 | 0.8437 | 0.7499 | 0.568 | 0.764 | 0.880 | 19.94× | reused | 26,083,328 |
| 50,000 | hnsw | 400 | completed | 0.9996 | 0.8440 | 0.7500 | 1.381 | 1.805 | 2.071 | 8.17× | reused | 26,083,328 |

Min/max latency, per-class metrics and confusion matrices are retained in `sweep.csv`, `sweep.json`, `repetition_details.csv` and the database.

### Controlled HNSW diagnostic

All eight controlled runs completed, used the exact HNSW index, persisted the requested `applied_ef_search`, and were marked `planner_mode='controlled_hnsw'` / `valid_for_summary=true`. Median p50 latency was: 10k ef=10/50/100/400 → 0.282/0.397/0.499/1.104 ms; 50k → 0.288/0.417/0.590/1.415 ms. These values are **not** pooled with natural results.

## 4. Provenance and access-path verification

- Both variants were created with seed 0; selected training ids and their SHA-256 were exported in `variant_ids.json` and recomputed successfully from `fv_116` joined by `row_index`.
- Derived fingerprints and preprocessing-state hashes in `variant_ids.json` match the persisted `config.extra.pool_variant` of the experiments.
- Query ids are identical across all configurations within each size; all 3,000 ids per experiment are exported in `query_samples.json`.
- HNSW index parameters are `m=16`, `ef_construction=64` for every HNSW run.
- `hnsw.ef_search` was confirmed by `SHOW` on the executing connection and persisted in `environment.applied_ef_search`.
- Plan diagnostics, including detailed EXPLAIN text, index verification and per-query plan-check counts, are persisted in `environment` for every experiment.

## 5. Artifacts

All artifacts are under `docs/results/phase3/`:

- `sweep.csv`, `sweep.json`
- `repetition_details.csv`, `repetition_summary.csv`
- `query_samples.json`, `benchmark_orders.json`, `variant_ids.json`
- `controlled_diagnostic.json`
- `validation_summary.json`
- `standard_verifier_output.json`

### Validation status

A custom option-A/4-ef validator passed with **zero errors**: 30 natural + 8 controlled experiments; 342,000 latency rows; 114 repetition rows; no orphan rows; unique configuration keys; provenance hashes matched; parent fingerprint unchanged; historical experiments unchanged.

The repository's `scripts/verify_results.py` reports 18 errors and is retained verbatim in `standard_verifier_output.json`. Two causes are known and unrelated to data corruption:

1. it hard-codes the older six-ef matrix (`10,25,50,100,200,400`) although the approved protocol uses `10,50,100,400`;
2. it requires every HNSW row to be `completed` with an index scan, while option A explicitly requires the 10k/ef=400 natural failures to be preserved as failed diagnostics.

The verifier was not modified, and no raw artifact was altered to satisfy it.

## 6. Resource consumption

| Run | Experiments | Total runtime (s) | Retrieval wall time (s) | New index build time (s) |
|---|---:|---:|---:|---:|
| Natural | 30 | 1,165.9 | 957.4 | 5.0 |
| Controlled diagnostic | 8 | 254.0 | 182.3 | 0.0 (index reused) |
| **Total** | **38** | **1,419.9 s (23.7 min)** | **1,139.7 s** | **5.0 s** |

Database row growth: +2 datasets, +38 experiments, +342,000 latency rows, +114 repetition rows. Final counts: 8 datasets, 240 experiments, 961,840 latency rows, 581 repetition rows.

## 7. Database integrity

- `knn_knn_platform` used throughout; original dataset 116 and its vector tables unchanged.
- Experiments 382/383/384 remain exactly as before the run.
- Historical results, Phase 1/2 reports and the `phase3-preflight-unrelated` stash were not modified.
- No destructive migration, deletion or backfill was performed.

## 8. Acceptance decision

**Execution is complete and all required configurations are accounted for**, with raw measurements, diagnostics and provenance preserved.

**NO-GO for treating all natural cells as valid scientific results:** the three 10k/ef=400 natural HNSW runs failed their access-path check, as approved in option A. They are excluded from valid HNSW summaries and speed-ups.

**GO for the validated subset:** 27 natural experiments (3 exact + 24 valid HNSW) and 8 controlled diagnostics have valid access paths and complete measurements, subject to the independent statistical audit requested in Prompt 6.

Statistical analysis and thesis generation were **not** started. Stop here as required.
