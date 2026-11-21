# Phase 1 — Critical Project Stabilization: Implementation Report

**Branch:** `phase1-stabilization`  
**Date:** 2026-11-21  
**Database:** not queried, not modified  
**Docker:** not used; no Compose/Dockerfile exists in the repository

## A. Implementation summary

### Task A — Dataset API response contracts

**Problem found.** `DatasetSummary` required `preprocessing_mode`, but both dataset endpoints omitted it from their SQL projections and model construction.

**Root cause.** The SQL queries were written before `preprocessing_mode` became a required response field.

**Files changed.**

- `backend/app/api/datasets.py`
- `backend/tests/test_dataset_api_contract.py` (new)

**Solution.** Both `list_datasets` and `get_dataset` now select `preprocessing_mode` from `datasets` and pass it into `DatasetSummary`. A regression test monkeypatches the DB helpers, asserts the field is present in the SQL and returned on both endpoints, and checks the 404 path. `frontend/types/api.ts` already declared `preprocessing_mode`, so no frontend change was needed. Frontend type-checking passed.

**Remaining limitation.** The endpoint tests are unit tests with mocked DB helpers; they do not prove a live database round-trip.

### Task B — Hard-coded database credentials

**Problem found.** A usable password was present as a `Settings` default, in `.env.example`, and in `scripts/verify.bat`.

**Root cause.** Local development credentials were copied into tracked configuration.

**Files changed.**

- `backend/app/core/config.py`
- `backend/.env.example`
- `scripts/verify.bat`
- `README.md`
- `backend/tests/test_config_security.py` (new)

**Solution.** `Settings.postgres_password` is now a required Pydantic `SecretStr`, populated from `POSTGRES_PASSWORD`. Empty values are rejected. `database_url` and `admin_database_url` read the secret through `get_secret_value()` and URL-encode it. `.env.example` contains only a clearly marked placeholder. `verify.bat` requires `PGPASSWORD` from the environment instead of embedding a value. The untracked local `backend/.env` was left untouched.

**Remaining limitation / required manual action.** Because the old password may have been used in a real environment, it should be rotated outside this repository. No credential value is reproduced in this report.

### Task C — 10k preprocessing methodology

**Problem found.** The runner reduced `fv_<id>` after ingestion but continued using preprocessing state fitted on the full 50k training pool. Test queries were also transformed with that full-pool state.

**Root cause.** Preprocessing state was stored once per ingested dataset, while pool size was represented by physically deleting rows from the single vector table.

**Files changed.**

- `backend/app/services/preprocessing.py`
- `backend/scripts/run_experiments.py`
- `README.md`
- `docs/architecture.md`
- `docs/essential-steps.md`
- `backend/tests/test_preprocessing_variants.py` (new)

**Solution.** No schema change was needed. Each requested size is now materialised as a **derived dataset** using the existing `datasets` + `fv_/tv_` architecture:

1. Select the exact size-specific training rows with the existing largest-remainder allocator and seeded within-class sampling.
2. Reconstruct numeric/binary feature values from stored pgvector literals using the parent preprocessing state.
3. Fit a new preprocessing state on the selected training rows only.
4. Transform both the size-specific training rows and the held-out query rows with that same state.
5. Create a new dataset row and vector tables, with the new state stored in `datasets.preprocessing`.
6. Record the derived dataset id, parent id, target/actual size, seed, state source and fingerprint in each experiment's `config` JSON and in `benchmark_orders.json`.
7. Leave the original dataset and its vector tables unchanged.

Datasets with categorical one-hot blocks are rejected for size variants: raw category values are not stored and cannot be reconstructed reliably. For the primary Covertype pipeline (numeric + binary features only), the inverse is safe enough for refitting.

**Remaining limitations.** This code path was not executed against the local database in this phase. The inverse reconstruction depends on the existing `%.8g` vector text precision. Rerunning the benchmark will create new derived dataset rows and tables; those rows must be retained because deleting them would cascade-delete their experiments.

### Task D — Authoritative thesis

**Problem found.** `Thesis-Final-Verified.docx` matched v3, while `Thesis-KNN-PostgreSQL-Platform.docx` described an older 1,000-query benchmark. README pointed students to both and regenerated the older one.

**Files changed.**

- `docs/Thesis-KNN-PostgreSQL-Platform.docx` → `docs/archive/Thesis-KNN-PostgreSQL-Platform.docx` (git move; contents preserved)
- `README.md`
- `docs/essential-steps.md`
- `docs/tools/build_thesis.py`
- `docs/tools/build_guide.py`

**Solution.** `Thesis-Final-Verified.docx` is now the sole authoritative v3 thesis. The older document is archived and labelled superseded. README regeneration commands call `build_final_thesis.py`; `build_thesis.py` is documented as a legacy builder writing to `docs/archive/`.

**Remaining limitation.** Thesis result text was not rewritten, as required. New benchmark numbers are not yet available.

### Task E — Result verification language

**Problem found.** Documentation implied the verifier independently validated scientific results.

**Files changed.**

- `README.md`
- `docs/results/README.md`

**Solution.** Wording now says *artifact integrity check*. The docs explicitly state what `verify_results.py` checks (headers, counts, ids, seeds, plan flags, CSV/JSON/DB agreement) and what it does not check (recomputation of accuracy, Recall@K or latency percentiles from raw per-query evidence).

**Remaining limitation.** An independent scientific recomputation script is still a Phase 2 item.

## B. Test report

### Commands executed

```bash
python ... -c "ast.parse(...)"                  # syntax parse of changed Python files
cd backend
.venv/Scripts/python.exe -m pytest -p no:cacheprovider -q \
  tests/test_dataset_api_contract.py \
  tests/test_config_security.py \
  tests/test_preprocessing_variants.py
.venv/Scripts/python.exe -m pytest -p no:cacheprovider -q \
  tests/test_api_validation.py \
  tests/test_benchmark_order.py \
  tests/test_classification.py \
  tests/test_datasets.py \
  tests/test_environment.py \
  tests/test_preprocessing.py \
  tests/test_tie_aware_recall.py \
  tests/test_dataset_api_contract.py \
  tests/test_config_security.py \
  tests/test_preprocessing_variants.py
cd ../frontend && ./node_modules/.bin/tsc --noEmit
git diff --check
```

### Results

| Check | Result |
|---|---|
| Targeted new tests | 9 passed, 0 failed, 0 skipped |
| Database-free backend subset | 148 passed, 0 failed, 0 skipped |
| Frontend `tsc --noEmit` | passed, no output |
| Python syntax parse | all changed Python files OK |
| `git diff --check` | passed after LF normalization |

One non-failing warning was emitted by Starlette about `httpx`/`starlette.testclient` deprecation.

### Tests not executed

- Full backend pytest suite: many tests use a live PostgreSQL + pgvector database and create/delete synthetic datasets and experiments.
- Integration tests in `test_vector_store.py`, `test_run_experiments.py`, `test_training_subsample.py`, `test_timing_accounting.py`, `test_dataset_stats.py`, `test_exact_tie_breaking.py`, `test_query_sampling.py` and `test_tie_tolerance_metrics.py`.
- `scripts.verify_results`: reads the live experiment database.
- Full benchmark: explicitly out of scope.

## C. Database impact report

- **Database modified:** No. No SQL was executed against the application database during this phase.
- **Migrations created:** No. No schema file was changed.
- **Existing data changed:** No. Existing experiment rows, artifacts and vector tables are untouched.
- **Result artifacts changed:** Only `docs/results/README.md` explanatory text was edited. CSV/JSON result files were not touched.
- **Future-run impact:** When the new runner is executed, it will add derived dataset rows, `fv_/tv_` tables and experiment rows. It will not truncate or overwrite the original dataset.
- **Manual action needed:** Back up the database before the next benchmark run. Rotate the previously exposed password outside the repository.

## D. Scientific impact report

The correction affects **cross-pool comparisons**, not every result.

- Within a single pool, exact and HNSW still share the same training vectors, query vectors, metric, `k` and voting code. The exact-vs-approximate comparison remains structurally valid.
- The old 10k arm used scaling statistics computed from the full 50k training split. That was not held-out-test leakage, but it did give the 10k arm information from 40k rows outside its own training pool.
- Therefore the old “10k vs 50k” scaling interpretation is affected: the two arms differed in preprocessing fit set as well as retrieval pool size.
- The `ef_search` trade-off within each pool, the retrieval-vs-classification asymmetry, and the index-plan verification claims are not directly invalidated.
- New numbers should be generated under a new run id before updating thesis tables or figures. Do not mix old and new artifacts.

## E. Next-phase recommendations

1. Approve the derived-dataset approach and its future database writes.
2. Back up the local database before any rerun.
3. Run a small isolated smoke benchmark using a synthetic or derived dataset, not the full benchmark.
4. Verify per experiment that `config.extra.pool_variant.state_source == "refit_on_variant_training_pool"`.
5. Add an independent recomputation script for metrics from raw per-query rows.
6. Rerun 10k and 50k benchmarks under a new run id only after smoke validation.
7. Regenerate figures and the authoritative thesis only after the new artifacts pass integrity checks.
8. Document the categorical-dataset limitation if the platform is used beyond Covertype.
