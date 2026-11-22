# Phase 2 — Controlled Smoke Test Report

**Branch:** `phase1-stabilization`  
**Run ID:** `smoke-phase2-20261121_194103`  
**Outcome:** **FAILED — do not proceed to the full benchmark**  
**Full benchmark:** not run  
**Thesis builder:** not run  

## 1. Backup verification

| Item | Value |
|---|---|
| Database | `knn_knn_platform` |
| Host / port | `localhost:5432` |
| PostgreSQL version | 18.0 |
| Confirmed not `test_phase11` | yes |
| Backup directory | `C:\Users\aria\Documents\ml-semester-project-db-backups` |
| Backup file | `knn_knn_platform_20261121_193852.dump` |
| Backup timestamp | 2026-11-21 19:38:52 local |
| File size | 74,568,125 bytes |
| SHA-256 | `be3ec348a058875736643adf67432cb0be0aeef0f96cf34165474b999c2fc9f5` |
| `pg_dump -Fc` | exit 0 |
| `pg_restore --list` | exit 0, 15 `TABLE DATA` entries |
| Temporary restore | exit 0 into `phase2_restore_20261121_193852` |
| Restored counts | matched source counts |
| Temporary restore DB | dropped after verification |

Source/restored counts were identical: `datasets=5`, `experiments=199`, `query_latencies=619720`, `experiment_repetitions=464`, `hnsw_indexes=0`.

## 2. Smoke workload

| Item | Value |
|---|---|
| Parent dataset | 116 (`Covertype 50k`) |
| Derived dataset | 846 |
| Training rows | 1,000 |
| Query rows in derived test table | 3,000 |
| Evaluated queries | 40 fixed stratified ids |
| Repetitions | 1 |
| `k` | 5 |
| Metric | L2 |
| Configurations | exact, HNSW `ef=10`, HNSW `ef=50` |
| Rounds | 1 |

## 3. Experiment execution

| Experiment | Method | ef | Status | Index scan | Sequential scan | Plan checked |
|---:|---|---:|---|---|---|---|
| 382 | exact | — | completed | false | true | true |
| 383 | HNSW | 50 | completed | **false** | **true** | true |
| 384 | HNSW | 10 | completed | true | false | true |

**Critical failure:** experiment 383 (`HNSW ef=50`) reported `used_sequential_scan=True` and `used_index_scan=False`. Its latency is an exhaustive-scan measurement, not an HNSW index measurement. The runner emitted the expected warning.

A plan check was executed for every configuration, but for `ef=50` the check revealed an unacceptable access path.

## 4. Main metrics

| Method | ef | Recall@5 | Accuracy | Macro F1 | p50 ms |
|---|---:|---:|---:|---:|---:|
| exact | — | 1.0000 | 0.6000 | 0.2571 | 0.41195 |
| HNSW | 50 | 1.0000 | 0.6000 | 0.2571 | 0.39710 |
| HNSW | 10 | 1.0000 | 0.6000 | 0.2571 | 0.36765 |

These are smoke-test observations on 40 queries with one repetition. They are **not** scientific results and must not be used for the thesis.

## 5. Provenance verification

| Check | Result |
|---|---|
| Derived dataset id | 846 |
| Parent dataset id in `benchmark_orders.json` | 116 |
| Experiments referencing dataset 846 | 3 of 3 |
| Selected training ids | exported once, 1,000 ids |
| Selected-ID hash | `dd3f86a18421df59b1089bcb6cb4fbb2cdebaf22ccab6536b51fd73d627eae82` |
| Recomputed selected-ID hash | matches |
| Fitted-state hash | stored and recomputed successfully |
| Derived fingerprint | stored and recomputed successfully |
| Exact ids absent from experiment config | confirmed |
| CSV/JSON agreement | 3 rows, ids match |
| Run ID agreement | all artifacts and DB rows use `smoke-phase2-20261121_194103` |

Retrieved neighbour ids were recorded for all 40 queries in all 3 experiments (`40/40` rows with non-empty `retrieved_ids`). Labels are not stored directly in `query_latencies`, but are recoverable from `fv_846` by neighbour id.

## 6. Original-dataset integrity

| Check | Before | After |
|---|---:|---:|
| Parent fingerprint | `6baa8fa283ea5a8267d3e964eedf5d2a` | unchanged |
| Parent train rows | 50,000 | 50,000 |
| Parent test rows | 3,000 | 3,000 |

No original dataset, vector table, experiment row or historical result was modified or deleted.

## 7. Generated artifacts

Directory:

`C:\Users\aria\Desktop\NextJS Projects\ml-semester-project\docs\results\smoke\phase2_20261121_194103`

Files:

- `sweep.csv`
- `sweep.json`
- `repetition_details.csv`
- `repetition_summary.csv`
- `query_samples.json`
- `benchmark_orders.json`
- `variant_ids.json`

Backup metadata:

`C:\Users\aria\Documents\ml-semester-project-db-backups\knn_knn_platform_20261121_193852.metadata.json`

## 8. Verification summary

Passed:

- backup file, checksum, archive listing and temporary restore;
- derived dataset creation with 1,000 rows;
- preprocessing provenance and hashes;
- original-dataset integrity;
- artifact/DB run-id consistency;
- neighbour-id recording;
- exact search path;
- HNSW index path for `ef=10`;
- no partial or orphaned derived objects.

Failed:

- HNSW `ef=50` used a sequential scan instead of the HNSW index.

## 9. Failures and unresolved issues

The smoke test did **not** pass its critical access-path check. The likely context is the very small 1,000-row training pool, but the cause has not been established and no fix was attempted.

Do not proceed to the full benchmark. The next safe step is to inspect the `EXPLAIN` output for the `ef=50` configuration on a derived pool and decide whether the smoke pool size must be increased or the planner behaviour must be addressed. No scientific conclusions are drawn from this run.
