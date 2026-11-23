# Phase 2 — Root-Cause Analysis

**Status:** investigation complete; no fix implemented  
**Audited commit:** `3dd6299` (`phase 2`) on `phase1-stabilization`  
**Database:** `knn_knn_platform` on localhost:5432, PostgreSQL 18.0, pgvector 0.8.7  
**Scope:** read-only source audit and live PostgreSQL execution-plan evidence  
**Recommendation:** **NO-GO** for the full benchmark; **GO** for the minimal validity guard and a controlled diagnostic re-test.

---

## 1. Repository-state gate

| Check | Expected | Observed | Action |
|---|---|---|---|
| Branch | `phase1-stabilization` | `phase1-stabilization` (tracks `origin/phase1-stabilization`) | matched |
| HEAD | `fc4898c` (`phase 1.1`) | `3dd6299` (`phase 2`, 2026-11-22) | mismatch reported; `fc4898c` is its direct parent |
| Working tree | clean | `M .workbuddy-ai/memory/2026-11-22.md`; `?? docs/ONE-PAGE-PROPOSAL.md` | reported; both left untouched |
| Phase 1 changes | present | present | confirmed |

`fc4898c..3dd6299` contains only `docs/PHASE2-SMOKE-REPORT.md`, the smoke result artifacts and one memory-log line. **No source file changed.** With explicit user approval, the investigation proceeded against `3dd6299`. No checkout, reset, rebase, amend, source edit, dataset edit, historical-result edit or destructive migration was performed.

## 2. Source audit

Files inspected at HEAD: `backend/app/services/vector_store.py`, `backend/app/services/experiments.py`, `backend/app/core/database.py`, `backend/app/db/schema.py`, `backend/scripts/run_experiments.py`, `backend/tests/test_vector_store.py`, `backend/tests/test_run_experiments.py`, `backend/tests/test_timing_accounting.py`, and the Phase 1 stabilization tests (`test_database_transaction.py`, `test_variant_lifecycle_integration.py`, `test_preprocessing_variants.py`, `test_dataset_api_contract.py`, `test_training_subsample.py`, `test_config_security.py`).

### HNSW execution path

1. **Index existence verification.** `ensure_hnsw_index()` checks the `hnsw_indexes` registry and `hnsw_index_exists()` queries `pg_indexes` for `fv_<id>_embedding_hnsw_idx` (`vector_store.py:322-364,458-466`). In the smoke run experiment 383 built the index (`index_reused=False`, `index_build_ms=117.34`, `index_size_bytes=532480`); experiment 384 reused it. Both configurations had a real, valid index.
2. **Index creation / operator class.** `_build_hnsw_index()` issues `CREATE INDEX ... USING hnsw (embedding vector_l2_ops) WITH (m=16, ef_construction=64)` (`vector_store.py:384-445`). Live catalog confirms `indexdef = CREATE INDEX fv_846_embedding_hnsw_idx ON public.fv_846 USING hnsw (embedding vector_l2_ops) WITH (m='16', ef_construction='64')`, `indisvalid/indisready/indislive = true`.
3. **ANALYZE.** `run_experiment()` calls `vs.analyze()` for both methods (`experiments.py:242,248`), which runs `ANALYZE fv_846` and `ANALYZE tv_846` (`vector_store.py:469-474`). Current `pg_stat_user_tables` shows `last_analyze = NULL`, `n_live_tup = 0` for `fv_846` (statistics snapshot appears to have been reset), but `pg_class.reltuples = 1000` equals the actual row count of 1,000. Statistics are not the cause.
4. **ef_search configuration.** `retrieve()` executes `SET hnsw.ef_search = <n>` and immediately `SHOW hnsw.ef_search` **on the same borrowed connection** used for the timed query and the EXPLAIN, then `RESET` in `finally` (`vector_store.py:627-660`). The pool is autocommit with `min_size=2`, `max_size=10` (`database.py:20-33`). A separate probe confirmed `SHOW` returns 10/50 and no leakage after `RESET`.
5. **SQL generation.** HNSW uses the index-compatible operator form `embedding <-> %s` with `ORDER BY embedding <-> %s LIMIT 5`; the exact baseline uses the function form `l2_distance(embedding,%s::vector)` plus an `id` tiebreak (`vector_store.py:536-577`). Live SQL captured from the application's own helpers is:

   ```sql
   SELECT id, label, embedding <-> %s AS distance FROM "fv_846" ORDER BY embedding <-> %s LIMIT 5
   ```
6. **Query execution.** One warm-up plus the timed query run on the same connection (`vector_store.py:641-648`).
7. **EXPLAIN verification.** `_explain_sql()` runs `EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT)` on that same connection while ef is still set; `plan_uses_index_scan(plan, index_name(dataset_id))` requires the *exact* HNSW index name (`vector_store.py:580-596,495-507`). `retrieve_batch()` verifies the first query of the first repetition only (`vector_store.py:759-781`). This is the same SELECT shape as the timed query.
8. **Result persistence.** `used_index_scan`, `environment.applied_ef_search`, `environment.used_sequential_scan` and `environment.plan_verified` are persisted, but **the EXPLAIN text itself is not**; only a boolean survives. `status` is hard-coded to `"completed"` (`experiments.py:482`).
9. **CSV/JSON export.** `flatten()` adds `used_sequential_scan` and `applied_ef_search` from `environment`; `print_warning_if_index_unused()` emits the warning observed in the smoke test (`run_experiments.py:681-707`). All three smoke rows were exported, including the invalid one.

### Differences from the previously uploaded source (`e8dc794`)

Unchanged from `e8dc794`: `vector_store.py`, `experiments.py`, `schema.py`, `test_vector_store.py`, `test_run_experiments.py`, `test_timing_accounting.py`. Changed are `app/core/database.py` (added `transaction()`), `app/core/config.py`, `app/services/preprocessing.py`, `app/api/datasets.py`, `scripts/run_experiments.py` (atomic variant lifecycle, provenance hash, `--variant-ids`, guarded legacy helpers), plus new `reproducibility.py` and Phase 1 tests. Therefore the retrieval and plan-verification code that produced the anomaly is the same code as in the uploaded snapshot; the anomaly is not a recent Phase 1 regression.

## 3. Evidence collection

Identical database state and query vector were used for both ef values: derived dataset 846, 1,000 training rows, 3,000 query rows, query `id = 24` (the first id in the 40-query set shared by experiments 382/383/384, confirmed identical). The exact application SQL shape was used; only `hnsw.ef_search` changed. No `enable_*` GUC was touched in the natural runs. Forced mode used `BEGIN; SET LOCAL enable_seqscan = off; ... ROLLBACK;` after natural behavior had been recorded, and a fresh connection confirmed `enable_seqscan=on`, `hnsw.ef_search=40` afterwards.

### Index, table and planner inputs

| Item | Value |
|---|---|
| Index | `fv_846_embedding_hnsw_idx`, access method `hnsw`, opclass `vector_l2_ops`, valid/ready/live, `m=16`, `ef_construction=64` |
| Table rows / `reltuples` / `relpages` | 1,000 / 1,000 / 44 |
| Index `reltuples` / `relpages` | 1,000 / 65 |
| Heap / total table size | 360,448 B (352 kB) / 983,040 B |
| Index size | 532,480 B (520 kB) |
| Relevant settings | `enable_seqscan=on`, `enable_indexscan=on`, `random_page_cost=4`, `seq_page_cost=1`, `effective_cache_size=4GB`, `work_mem=4MB`, `max_parallel_workers_per_gather=2`, `jit=on`, `plan_cache_mode=auto` |
| ef on executing connection | `SHOW hnsw.ef_search` → `10` and `50` respectively; `applied_ef_search` in experiments 384/383 matches |

### ef=10 vs ef=50 (same vector, same table, same SQL)

| Field | ef=10 | ef=50 |
|---|---|---|
| Access path | `Index Scan using fv_846_embedding_hnsw_idx` | `Seq Scan on fv_846` + top-N `Sort` |
| Index Searches | 1 | n/a |
| Limit startup..total cost | 44.08..46.15 | 75.61..75.62 |
| Child-node cost | index scan 44.08..458.50 | sort 75.61..78.11; seq scan 0.00..59.00 |
| Actual execution time (app-shape EXPLAIN) | 0.127 ms | 0.280 ms |
| Shared buffers | hit=150 | hit=44 |
| Verbose/SETTINGS run execution | 29.808 ms, hit=101 read=52 (first, cold-buffer run) | 0.295 ms, hit=44 |
| `hnsw.ef_search` shown | 10 | 50 |
| Status persisted by current code | `completed` | `completed` (**invalid**) |

The 29.8 ms figure is a single cold-buffer diagnostic, not an HNSW benchmark result; warm-buffer values are shown for comparability. Neither is a scientific latency measurement.

`EXPLAIN (ANALYZE, BUFFERS, VERBOSE, SETTINGS)` is supported on PostgreSQL 18; no `Settings:` block appeared because no standard planner GUC differed from its default. The ef value is independently established by `SHOW` on the executing connection and by the ef-dependent cost change.

### Natural planner sweep — 1,000-row pool (`fv_846`)

| ef_search | access path | top-node cost |
|---|---|---|
| 10 | HNSW index scan | `Limit  (cost=44.08..46.15 rows=5 width=26)` |
| 20 | HNSW index scan | `Limit  (cost=67.27..69.23 rows=5 width=26)` |
| 25 | Seq Scan | `Limit  (cost=75.61..75.62 rows=5 width=26)` |
| 30 | Seq Scan | `Limit  (cost=75.61..75.62 rows=5 width=26)` |
| 40 | Seq Scan | `Limit  (cost=75.61..75.62 rows=5 width=26)` |
| 50 | Seq Scan | `Limit  (cost=75.61..75.62 rows=5 width=26)` |
| 75 | Seq Scan | `Limit  (cost=75.61..75.62 rows=5 width=26)` |
| 100 | Seq Scan | `Limit  (cost=75.61..75.62 rows=5 width=26)` |
| 200 | Seq Scan | `Limit  (cost=75.61..75.62 rows=5 width=26)` |

The crossover is between ef=20 and ef=25. The sequential-scan-plus-sort alternative has a constant total plan cost of 75.61; the HNSW index startup cost rises with ef_search.

### Forced HNSW mode — 1,000-row pool (diagnostic only, transaction-local, rolled back)

| ef_search | Limit node cost with forced HNSW |
|---|---|
| 10 | `Limit  (cost=44.08..46.15 rows=5 width=26)` |
| 50 | `Limit  (cost=127.96..129.61 rows=5 width=26)` |
| 100 | `Limit  (cost=217.83..219.03 rows=5 width=26)` |
| 200 | `Limit  (cost=267.50..268.45 rows=5 width=26)` |

At ef=50 the forced HNSW index has startup cost 127.96, i.e. **higher** than the 75.61 cost of the naturally chosen sequential plan. This proves the planner is choosing the cheaper plan, not failing to consider the index.

### Natural planner on an existing 50,000-row indexed pool (`fv_1`, read-only)

| ef_search | access path | Limit node cost |
|---|---|---|
| 10 | HNSW index scan | `Limit  (cost=67.14..69.21 rows=5 width=26)` |
| 50 | HNSW index scan | `Limit  (cost=195.97..198.03 rows=5 width=26)` |
| 100 | HNSW index scan | `Limit  (cost=334.00..336.04 rows=5 width=26)` |

With 50,000 rows the sequential alternative is far more expensive, so index scans remain natural at ef=10/50/100. This is a different dataset and query vector; it supports the cost mechanism but is not a substitute for a controlled smoke re-test at the planned pool sizes.

## 4. Root cause

**Established.** In pgvector 0.8.7 the HNSW access-path cost estimate increases with `hnsw.ef_search`. On a 1,000-row table the HNSW startup cost is 44.08 at ef=10, 67.27 at ef=20 and 127.96 at ef=50, while the exhaustive `Seq Scan` + top-N `Sort` path costs 75.61 regardless of ef. At ef=50 PostgreSQL therefore correctly selects the cheaper sequential plan. The experiment runner detects this and warns, but the service still records the run as `completed`, so the sequential-scan latency is exposed as if it were an HNSW result.

Ruled out by evidence: missing or invalid index (valid, right AM/opclass, correct build for L2), SQL incompatibility (operator form matches HNSW), wrong distance operator (`<->` with `vector_l2_ops`), session/connection leakage (`SHOW` and `RESET` verified on the same connection), stale statistics (reltuples exactly 1,000), and a recent Phase 1 source change (retrieval code unchanged since `e8dc794`).

## 5. Remaining uncertainties / alternatives not fully resolved

1. Pool sizes 5,000 / 10,000 / 20,000 were not measured because no existing indexed pools of those sizes are present; creating them would modify the production database, which is forbidden here. Only 1,000 and an existing 50,000-row pool were measured.
2. The exact internal pgvector cost formula was inferred from observed costs, not read from pgvector C source.
3. The current `pg_stat_user_tables` snapshot has no `last_analyze`; it may have been reset since the smoke run. Evidence still shows accurate `reltuples`.
4. `plan_cache_mode=auto` was not separately tested for a generic-vs-custom plan transition after repeated prepared executions; all captured plans nevertheless match the persisted access-path flags.
5. Cold versus warm buffer state changes measured time (29.8 ms vs 0.127 ms at ef=10) but not plan choice; no cold-cache benchmark procedure exists.

A safe diagnostic plan for (1) is to create derived, disposable pools at the planned sizes in an **isolated** PostgreSQL database, build the same L2 HNSW index, run the same natural EXPLAIN sweep at ef=10/50/100, and record the crossover before any benchmark decision.

## 6. Validity and persistence audit — Phase D

**Yes, an HNSW experiment can currently be persisted with `status='completed'` while `used_index_scan=False`.** Experiment 383 is a real example. The assignment at `experiments.py:482` is unconditional; `used_index_scan` is informative metadata only. `compare_experiments()`, `list_experiments()` and the runner CSV/JSON export do not filter invalid HNSW rows.

**Minimal correction (not implemented):**

1. In `run_experiment()`, compute status from the plan check: `method == 'hnsw' and not used_index_scan` → `status='failed'` (existing enum), with an `error_message` containing dataset id, ef, expected index name and the observed path. All other paths, including exact search, keep `status='completed'`.
2. Persist the EXPLAIN text and verified index name in the existing `environment` JSONB (e.g. `execution_plan`, `verified_index_name`) — no schema migration required. This preserves diagnostics currently lost after the run.
3. Filter valid scientific summaries by `status='completed'` (defence in depth: also require `used_index_scan IS NOT FALSE` for HNSW) in comparison, export and thesis-input code. Invalid runs remain in the database and may be exported to a separate diagnostics artifact; do not delete or rewrite experiment 383.
4. Add the regression tests below. No destructive migration, no historical backfill, no change to exact-baseline behavior, no API/schema break.

## 7. Regression-test plan — Phase E

All tests must use disposable datasets named `pytest-phase2-*` in an isolated database (`test_phase2`), clean up their own rows/tables/experiments, and never touch dataset 116, dataset 846 or historical experiment ids.

| # | Test | Assertion |
|---|---|---|
| 1 | HNSW index selected | On a 20k-row synthetic pool, ef=10 plan contains the exact `fv_<id>_embedding_hnsw_idx`, `used_index_scan=True`, `used_sequential_scan=False`. |
| 2 | HNSW index not selected | On a 1k-row pool with a valid index, ef=50 plan is a seq scan; flags are false/true; after the fix, stored status is `failed` with an error message. |
| 3 | ef_search correctly applied | `SET`/`SHOW` on the executing connection returns 10 and 50; `applied_ef_search` persisted; index startup cost or neighbour sets differ as expected. |
| 4 | No ef leakage across pooled connections | Force pool `min_size=max_size=1`; run ef=400 then a no-ef query; `SHOW` on the *same* reused connection returns default 40; a second borrower also sees 40. |
| 5 | Small training pool | 1k pool: record natural fallback at ef>=25 and assert diagnostic status; do not assert index usage. |
| 6 | Larger training pool | 20k/50k isolated pool: ef=10/50/100 remain index scans under natural planner settings. |
| 7 | Exact remains exhaustive | With and without an HNSW index, exact uses the function form, seq scan, `id` tiebreak, `status='completed'`, `used_index_scan=False`. |
| 8 | Invalid HNSW cannot be accepted | Inject `used_index_scan=False` into a batch result; assert `status='failed'`, diagnostics retained, and the row is excluded from valid summaries/exports. |
| 9 | Correct index name in plan check | Unit test: a plan naming a btree index returns `False`; a plan naming the expected HNSW index returns `True`; integration test with both indexes present. |
| 10 | Phase 1 guarantees intact | Run the guarded Phase 1 suite against an isolated `test_phase2` database: transaction rollback/autocommit restore, variant hashes, immutability, destructive-helper guards, API contract. |

## 8. GO / NO-GO

**NO-GO for the full benchmark.** The root cause is established, but the current persistence contract still allows an exhaustive scan to be reported as a successful HNSW result (demonstrated by experiment 383), and the planned 5k/10k/20k pool behavior has not been measured under natural planner settings.

**GO for** implementing the minimal validity guard, persisting plan diagnostics, adding the regression tests, and running a controlled diagnostic smoke on isolated pools before any full benchmark or thesis use. Historical results, dataset 116 and experiment 383 must remain unchanged.
