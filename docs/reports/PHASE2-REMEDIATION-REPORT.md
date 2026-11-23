# Phase 2 — Minimal Remediation Report

**Branch:** `phase1-stabilization`  
**Audited/implementation HEAD:** `3dd6299` (`phase 2`)  
**Basis:** `docs/reports/PHASE2-ROOT-CAUSE-ANALYSIS.md`  
**Date:** 2026-11-23  
**Full benchmark:** not run  
**Thesis generation:** not run  
**Status:** implementation complete; approval requested before further work

---

## 1. Summary

The confirmed defect is corrected at the result-validity layer, not by silently forcing an index into the original benchmark.

- The original **natural planner mode** remains the default and still lets PostgreSQL choose any access path. Sequential-scan observations are preserved as diagnostics but are no longer accepted as valid HNSW measurements.
- A separate, explicitly opt-in **controlled HNSW mode** encourages index selection with transaction-local planner settings inside a real transaction, and verifies the actual index scan for every timed query.
- HNSW runs whose plan is not the verified index scan are persisted with `status='failed'`, an `error_message`, and full plan/index diagnostics in the existing JSONB `environment` column.
- No database migration was required. No historical record, dataset 116, dataset 846, or production experiment row was changed.

## 2. Files changed

| File | Change | Reason |
|---|---|---|
| `backend/app/services/vector_store.py` | Added `PLANNER_MODES` (`natural`, `controlled_hnsw`), catalog-level `verify_hnsw_index()` with `IndexVerification`, ef_search helpers that `SET`/`SHOW` on the executing connection and restore the previous value, and per-query plan checking. Refactored retrieval into `_retrieve_on_cursor()`; `retrieve_batch()` now runs controlled mode in one explicit `BEGIN ... SET LOCAL ... ROLLBACK` transaction. | Implements the two modes, proves ef was applied, verifies index existence/operator class, and ties a plan check to each timed query. |
| `backend/app/services/experiments.py` | Added `planner_mode` config field with validation; computes validity errors and sets `status`/`error_message`; stores plan text, index verification, ef, plan-check counts and validity flags in `environment`; `compare_experiments()` now reports `valid`/`validity_errors` and suppresses all numeric deltas when invalid. | Prevents invalid HNSW runs from entering valid scientific summaries while retaining diagnostics. |
| `backend/scripts/run_experiments.py` | Added `--planner-mode` (default `natural`), exported `planner_mode`/`status`/`valid_for_summary` in artifacts, included mode in repetition/summary exports, skipped invalid and controlled rows in speed-up computation, improved the warning, and tolerated hand-built `argparse.Namespace` objects. | Keeps natural and controlled results in separate categories and keeps the runner backward-compatible. |
| `backend/app/api/schemas.py` | Added optional `planner_mode` to request and summary schemas; validator rejects `controlled_hnsw` without `method='hnsw'`. | Exposes the opt-in mode through the API without breaking existing clients (default `natural`). |
| `backend/app/api/experiments.py` | Passes `request.planner_mode` into `ExperimentConfig`. | Wires the API field through to the service. |
| `backend/scripts/verify_results.py` | Checks HNSW `status='completed'` and planner mode; skips comparison of columns absent from older archived artifacts. | Invalid rows now fail verification; historical artifacts are never rewritten. |
| `backend/tests/test_phase2_remediation.py` | New guarded integration/unit regression suite using only `pytest-phase2-*` datasets in a `test_` database. | Covers the confirmed defect, invalid access paths, ef isolation, index-name checks and Phase 1 guarantees. |

The pre-existing modification to `.workbuddy-ai/memory/2026-11-22.md` was left untouched.

## 3. Behavior of the two modes

### A. Natural planner mode (default)

- No `enable_*` GUC is changed.
- PostgreSQL chooses the plan; sequential scans remain observable and are preserved in `environment.execution_plan`.
- An HNSW run that does not use the verified HNSW index for **every** timed query is persisted as `status='failed'` with `error_message` and `valid_for_summary=false`.
- Exact search is unchanged: function-form distance, exhaustive scan, `status='completed'`.

### B. Controlled HNSW mode (opt-in)

- Selected with `--planner-mode controlled_hnsw`, `ExperimentConfig(planner_mode='controlled_hnsw')`, or the API field.
- Requires `method='hnsw'`.
- Uses one explicit transaction: `BEGIN`, `SET LOCAL enable_seqscan = off`, `SET LOCAL hnsw.ef_search = <n>`, timed queries, `ROLLBACK`. No `SET LOCAL` outside a transaction.
- Records `planner_mode='controlled_hnsw'` and `planner_settings.enable_seqscan='off'`.
- Still fails the run if the verified HNSW index is not used; encouragement is never mistaken for success.
- Speed-ups against the natural exact baseline are deliberately not computed for controlled rows.

The two modes are separate result categories. Nothing forces index usage in the original natural benchmark.

## 4. Verification details

- `hnsw.ef_search` is applied with `SET` (natural) or `SET LOCAL` (controlled) on the exact connection executing the query, read back with `SHOW`, and compared to the requested value; a mismatch raises before the timed query.
- The previous GUC value is captured before `SET` and restored in `finally` (or expires with the rolled-back transaction), preventing pooled-connection leakage.
- Index verification checks the catalog for the exact index name, `hnsw` access method, metric-compatible operator class (`vector_l2_ops` for L2), `embedding` column, and `indisvalid/indisready/indislive`.
- The detailed `EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)` is captured for the first query. Every timed HNSW query additionally gets a cheap `EXPLAIN (FORMAT TEXT)` of the same SQL shape after its timer stops, so no query is re-executed and no latency sample is contaminated.
- `used_index_scan` is true only when **all** checks name the verified index; `used_sequential_scan` is true if any check sees a sequential scan.

## 5. Test results

Tests ran against two newly created isolated local databases, `test_phase2` (backend settings and Phase 2 tests) and `test_phase1` (guarded Phase 1 lifecycle tests). Neither is the production database `knn_knn_platform`.

| Run | Result |
|---|---|
| Targeted Phase 2 + Phase 1 integration suite | **88 passed**, 1 warning (Starlette/httpx deprecation) |
| Full backend suite (`tests`) | **272 passed, 1 skipped**, 1 warning |
| Skipped test | `tests/test_tie_tolerance_metrics.py:154` — pre-existing skip: "cosine distance is discrete for collinear vectors" |
| `git diff --check` | passed |

The targeted suite included `test_phase2_remediation.py`, `test_vector_store.py`, `test_run_experiments.py`, `test_timing_accounting.py`, `test_preprocessing_variants.py`, `test_training_subsample.py`, `test_variant_lifecycle_integration.py`, `test_dataset_api_contract.py`, `test_database_transaction.py` and `test_config_security.py`.

### Regression coverage

| Required case | Test |
|---|---|
| HNSW index selected successfully | `test_controlled_hnsw_selects_index_at_small_pool` (1k pool, ef=50, controlled mode) |
| HNSW index not selected | `test_natural_fallback_is_failed_and_not_a_valid_summary` (1k × 54-dim pool, ef=50) |
| ef_search correctly applied | `test_ef_search_is_applied_and_restored_on_same_connection` |
| ef_search does not leak between pooled connections | same test: `SET`/`SHOW`/restore on one borrowed connection |
| Small training-pool behavior | `test_natural_fallback_is_failed_and_not_a_valid_summary` |
| Larger training-pool behavior | existing `test_vector_store.py` synthetic 20k-pool suite |
| Exact search remains exhaustive | `test_exact_search_stays_exhaustive_and_valid` |
| Invalid HNSW cannot be accepted as valid | natural fallback test asserts `status='failed'`, `valid_for_summary=false`, and `compare_experiments().valid is False` with all deltas `None` |
| Plan verification uses correct index name | `test_plan_check_uses_exact_index_name` |
| Phase 1 guarantees intact | `test_phase1_transaction_guarantee_intact` plus the full guarded Phase 1 suite |

## 6. Database and migration information

- **Migration:** none. `status='failed'` already exists in the schema CHECK constraint; diagnostics use the existing `environment` JSONB column; `error_message` already exists. No DDL was applied to the production database.
- **Production database check after implementation:** `knn_knn_platform` still has 6 datasets, 202 experiments and 619,840 latency rows. Experiments 382/383/384 are unchanged; experiment 383 remains `status='completed'` with `used_index_scan=false` because historical records were deliberately not backfilled.
- **Test databases:** `test_phase2` and `test_phase1` were created on the local PostgreSQL server for verification and contain only disposable test artifacts.

## 7. Git diff summary

```
 .workbuddy-ai/memory/2026-11-22.md   |   1 +   (pre-existing, untouched)
 backend/app/api/experiments.py       |   1 +
 backend/app/api/schemas.py           |   4 +
 backend/app/services/experiments.py  | 123 ++++++--
 backend/app/services/vector_store.py | 556 +++++++++++++++++++++++++++++------
 backend/scripts/run_experiments.py   |  50 +++-
 backend/scripts/verify_results.py    |   9 +
 7 files changed, 622 insertions(+), 122 deletions(-)
```

Untracked new files from this task: `backend/tests/test_phase2_remediation.py`, `docs/reports/PHASE2-REMEDIATION-REPORT.md`, and the earlier investigation reports in `docs/reports/`. `docs/ONE-PAGE-PROPOSAL.md` was already untracked before this task. No commit was created.

## 8. Remaining risks

1. Controlled HNSW mode encourages index use with `enable_seqscan=off`; it does not change pgvector's cost model. Natural planner fallbacks at small pools remain expected and are now correctly labelled invalid.
2. The cheap per-query `EXPLAIN` verifies the same SQL shape after execution; a pathological generic/custom plan-cache transition could still make that plan differ from the executed one. The first query also carries a detailed `EXPLAIN ANALYZE` to reduce this risk.
3. Natural behavior at 5k/10k/20k rows is still unmeasured. A controlled diagnostic re-test on isolated pools remains necessary before any full benchmark.
4. Historical invalid experiment 383 retains its original `completed` status by design. Consumers of old artifacts must use the new validity signals or continue to treat that smoke run as non-scientific.
5. The local `test_phase2` and `test_phase1` databases remain available for inspection; they were not dropped during this task.

## 9. Acceptance decision

**GO for the remediation itself:** the confirmed defect is corrected, the regression suite passes, invalid HNSW runs cannot enter valid summaries, historical data is intact, and Phase 1 guarantees remain intact.

**NO-GO for the full benchmark remains in force** until a controlled diagnostic re-test on isolated pools is reviewed and explicitly approved.

Implementation is stopped here for your approval. No fix beyond the changes listed above, no benchmark, and no commit has been made.
