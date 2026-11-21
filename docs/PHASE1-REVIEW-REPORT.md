# Phase 1 — Independent Review Report

**Review type:** read-only inspection and static verification  
**Branch:** `phase1-stabilization`  
**Base commit used for comparison:** `0da8db0`  
**Review date:** 2026-11-21  
**Database:** not queried or modified  
**Benchmark:** not executed  
**Source files:** not modified  
**Result artifacts:** not modified  

This report is the only file created by the review.

---

## Review 1 — Git and implementation scope

### 1.1 Git state

At inspection time:

- Current branch: `phase1-stabilization`.
- Working tree: clean.
- Recent commits:
  - `c92d620 phase 1 implemented`
  - `e62d4c0 phase 1`
  - `0da8db0 student guide` (base)

The implementation was committed before this review. I did not create a commit.

### 1.2 Changed files

`git diff 0da8db0..HEAD` reports 20 files:

| File | Change | Review classification |
|---|---|---|
| `README.md` | dataset setup note, verification wording, thesis authority, derived-variant explanation | related |
| `backend/.env.example` | credential value replaced by placeholder | related |
| `backend/app/api/datasets.py` | selects/returns `preprocessing_mode` | related |
| `backend/app/core/config.py` | required `SecretStr` password, URL encoding | related |
| `backend/app/services/preprocessing.py` | vector inverse helpers | related |
| `backend/scripts/run_experiments.py` | derived-variant creation and provenance | related, highest risk |
| `backend/tests/test_config_security.py` | new config tests | related |
| `backend/tests/test_dataset_api_contract.py` | new API contract tests | related |
| `backend/tests/test_preprocessing_variants.py` | new preprocessing helper tests | related |
| `docs/architecture.md` | derived-variant note | related |
| `docs/essential-steps.md` | thesis and variant documentation updates | related |
| `docs/results/README.md` | artifact-integrity wording | related |
| `docs/tools/build_guide.py` | companion thesis reference updated | related |
| `docs/tools/build_thesis.py` | legacy builder output path and docstring | related |
| `scripts/verify.bat` | password from environment | related |
| `docs/Thesis-KNN-PostgreSQL-Platform.docx` → `docs/archive/Thesis-KNN-PostgreSQL-Platform.docx` | archived thesis | related |
| `docs/AUDIT-REPORT.md` | prior-task audit deliverable | unrelated to Phase 1 implementation |
| `docs/PHASE1-IMPLEMENTATION-REPORT.md` | Phase 1 deliverable | related |
| `.workbuddy-ai/memory/2026-11-20.md` | internal memory note | unrelated |
| `.workbuddy-ai/memory/2026-11-21.md` | internal memory note | unrelated |

No experiment artifacts under `docs/results/*.csv` or `docs/results/*.json` were modified. `docs/results/README.md` was edited in text only.

### 1.3 Unrelated changes

Two internal memory files and the prior audit report were committed with the Phase 1 work. They do not affect runtime behavior, but they are unrelated to the five Phase 1 tasks and should have been left out of the implementation commit if the branch were intended to contain only code and Phase 1 documentation.

### 1.4 Thesis move

The old thesis was moved, not rewritten:

- Base blob: `2c33fc87941964631a6d3d53b1e858756480688b`
- Current archived blob: `2c33fc87941964631a6d3d53b1e858756480688b`
- Git status: `R100`

The content is preserved byte-for-byte.

`docs/tools/build_final_thesis.py::OUTPUT` is `docs/Thesis-Final-Verified.docx`, as intended. The builder was **not executed**. Inspection shows it:

- reads `docs/results/sweep.csv`;
- reads two row sets from the live database via `app.core.database.fetch_all`;
- writes six figures into `docs/results/figures/`;
- writes `docs/Thesis-Final-Verified.docx`.

It does not write source files, CSV/JSON result artifacts, or the archived thesis. However, it is not read-only: running it would overwrite the authoritative thesis and its six figure PNGs. That is acceptable only after a new verified benchmark, not during this review.

---

## Review 2 — Preprocessing variants

### 2.1 Complete execution trace

`scripts/run_experiments.py::main` loops sizes and calls `build_preprocessed_variant(parent_id, size, seed)` at lines 645–655.

Inside `build_preprocessed_variant`:

1. `fetch_one("SELECT * FROM datasets WHERE id = %s")` loads the parent dataset.
2. `PreprocessState.from_dict(parent["preprocessing"])` reconstructs the parent fitted state.
3. `_snapshot_pool(parent_id, "train")` and `_snapshot_pool(parent_id, "test")` read all `fv_/tv_` rows.
4. `_choose_subset_ids(train_rows, max_train_rows, seed)` selects exact stratified ids.
5. `selected_train` keeps only rows with those ids.
6. `inverse_transform_vectors(...)` reconstructs numeric/binary feature values from stored vector text.
7. `labels_to_original(...)` maps encoded labels back to original class values.
8. `fit_preprocessor(raw_train, ...)` fits the new state.
9. `transform(raw_train, ...)` and `transform(raw_test, ...)` produce matrices using that new state.
10. `_insert_variant_dataset(...)` creates a new `datasets` row holding the new state.
11. `vs.create_dataset_tables(variant_id, dimension)` creates `fv_variant` and `tv_variant`.
12. `vs.insert_vectors(...)` fills both tables.
13. `main` passes `dataset_id=variant_id` and `extra={"pool_variant": variant}` into `run_one`.
14. `experiments.run_experiment` loads `dataset["preprocessing"]` from the variant dataset row and reads query vectors from `tv_variant`.

### 2.2 Verification against the nine required checks

| # | Requirement | Finding | Evidence |
|---|---|---|---|
| 1 | Selected 10k rows are used to fit the 10k state | **Satisfied in code.** `selected_train` is built from `_choose_subset_ids` and passed to `fit_preprocessor`. | `run_experiments.py::build_preprocessed_variant`, lines 457–476 |
| 2 | 50k rows have their own state | **Satisfied in code.** If `total <= max_train_rows`, `_choose_subset_ids` returns all rows; a new variant row and state are still created. | `run_experiments.py::_choose_subset_ids`, lines 336–338; `build_preprocessed_variant`, lines 474–487 |
| 3 | Queries use the exact state fitted for their pool | **Satisfied in code.** `raw_test` is transformed by the same `preprocessor` as `raw_train`, then written to `tv_variant`; `run_experiment` reads those vectors. | `build_preprocessed_variant`, lines 483–503; `experiments.py::run_experiment`, lines 181–200 |
| 4 | Test data never used to fit preprocessing | **Satisfied.** Only `raw_train` is passed to `fit_preprocessor`; `raw_test` is transformed only. | `build_preprocessed_variant`, lines 474–484 |
| 5 | Label encoding and feature ordering consistent | **Satisfied with a guard.** `fit_preprocessor` uses the same column types/order, and `state.class_labels != base_state.class_labels` raises before table creation. | `preprocessing.py::fit_preprocessor`, `inverse_transform_vectors`; `build_preprocessed_variant`, lines 477–481 |
| 6 | Fingerprints identify transformed pool | **Satisfied.** `dataset_fingerprint(variant_id)` hashes count, ordered `(id,label)` and vector text checksum of the variant `fv_` table. | `dataset_stats.py::dataset_fingerprint`; `build_preprocessed_variant`, line 511 |
| 7 | `config.extra.pool_variant` and `benchmark_orders.json` contain provenance | **Mostly satisfied.** They include parent/derived ids, target/actual size, seed, mode, fingerprint and `state_source`. Missing: parent fingerprint, exact selected ids and a hash of the fitted state. | `build_preprocessed_variant`, lines 505–514; `run_experiments.py::main`, lines 660–686, 712–721 |
| 8 | Re-running cannot silently reuse incompatible state | **Satisfied.** Every call creates a new dataset id and refits state; there is no cache. Downside: duplicates accumulate. | `_insert_variant_dataset`, lines 407–436; `main`, line 649 |
| 9 | Failed creation cannot leave partial registration | **NOT satisfied — BLOCKER.** The `datasets` row is inserted and committed before vector tables/COPY. Any later exception leaves an orphan dataset row and possibly empty/partial `fv_/tv_` tables. | `core/database.py::get_pool` uses `autocommit=True`; `_insert_variant_dataset` line 436; no `try/except/cleanup` in `build_preprocessed_variant` lines 487–514 |
| — | Cleanup safe for original dataset | **Satisfied.** `main` no longer calls `restore_pool` or `subset_dataset`; original `fv_/tv_` are only read. Legacy destructive helpers remain but are not invoked by `main`. | `main`, lines 635–649; `restore_pool` lines 308–331; `subset_dataset` lines 357–398 |

### 2.3 Additional preprocessing observations

- `inverse_transform_vectors` rejects categorical one-hot datasets. This is a safety choice, not a silent fallback. Evidence: `preprocessing.py` lines 501–505.
- Inverse reconstruction uses the parent scaler to recover pre-scaling values, then refits. For numeric/binary Covertype this is valid within `%.8g` vector-text precision, verified by `test_preprocessing_variants.py`.
- If a continuous column in a small subset happened to contain only 0/1 values, `infer_column_types` could classify it as binary and leave it unscaled. Unlikely for Covertype, but not guarded.
- `state.class_labels` equality check protects against a subset missing a class.
- The derived state is stored in `datasets.preprocessing`, so `run_experiment` does not need a schema change.

---

## Review 3 — Database safety

### 3.1 SQL and lifecycle

Derived dataset creation executes these operations, in this order:

1. `INSERT INTO datasets (...) RETURNING id` — `_insert_variant_dataset`.
2. `CREATE TABLE fv_<variant_id> (...)` and `CREATE TABLE tv_<variant_id> (...)` plus two `row_index` btree indexes — `vector_store.create_dataset_tables`.
3. `COPY` into `fv_<variant_id>` and `tv_<variant_id>` — `vector_store.insert_vectors`.
4. During HNSW runs, `CREATE INDEX fv_<variant_id>_embedding_hnsw_idx` and `INSERT ... ON CONFLICT` into `hnsw_indexes` — `vector_store._build_hnsw_index`.
5. `run_experiment` inserts one `experiments` row, N `experiment_repetitions` rows and `repetitions × queries` `query_latencies` rows.

No SQL in this path updates or deletes rows in the original dataset tables.

### 3.2 Tables and rows created

| Object | Created by | Notes |
|---|---|---|
| `datasets` row | `_insert_variant_dataset` | New id, `source='derived'`, own preprocessing JSON |
| `fv_<variant_id>` | `create_dataset_tables` | `id`, `row_index`, `label`, `embedding vector(dim)` |
| `tv_<variant_id>` | `create_dataset_tables` | Same shape |
| `fv/tv_<id>_row_idx` | `create_dataset_tables` | Ordinary btree indexes |
| `fv_<variant_id>_embedding_hnsw_idx` | `_build_hnsw_index` | Created only for HNSW runs |
| `hnsw_indexes` row | `_build_hnsw_index` | Upsert on `dataset_id` |
| experiment rows | `run_experiment` | Includes derived `dataset_id` |

### 3.3 Foreign keys

From `backend/app/db/schema.py`:

- `hnsw_indexes.dataset_id → datasets(id) ON DELETE CASCADE`.
- `experiments.dataset_id → datasets(id) ON DELETE CASCADE`.
- `experiment_repetitions.experiment_id → experiments(id) ON DELETE CASCADE`.
- `query_latencies.experiment_id → experiments(id) ON DELETE CASCADE`.

Deleting a derived dataset would cascade-delete its experiments. No deletion code exists in the Phase 1 path.

### 3.4 Cleanup, orphans, duplicates, concurrency

- **Cleanup:** none for derived variants. `restore_pool` can `TRUNCATE` an original table, but `main` no longer calls it.
- **Orphan risk:** high. A failure after `INSERT INTO datasets` but before/during `CREATE TABLE` or `COPY` leaves a registered dataset without usable vector tables. Evidence: autocommit and absence of compensating cleanup.
- **Duplicate variants:** every run creates new rows; there is no uniqueness constraint on `(parent_id, target_size, seed)`. Repeated runs are reproducible but not idempotent.
- **Concurrent runs:** table names use the unique `datasets.id` sequence, so two concurrent runs should not collide on `fv_/tv_` names. They can create duplicate variants. Because each statement autocommits, partially created variants from one run are visible to another.

---

## Review 4 — Credentials

### 4.1 Findings

| Requirement | Finding | Evidence |
|---|---|---|
| `POSTGRES_PASSWORD` required | Satisfied. `postgres_password: SecretStr = Field(...)`. | `config.py`, line 36 |
| Empty rejected | Satisfied by validator. | `config.py`, lines 25–30 |
| Secret not serialized/logged | Satisfied in inspected code. `SecretStr` hides repr; tests assert this. `get_pool` logs only database name; `capture_environment` does not include password. | `config.py`; `tests/test_config_security.py`; `database.py::get_pool`; `environment.py::capture_environment` |
| URL encoding | Satisfied. `quote(..., safe="")` applied in both URL properties. | `config.py`, lines 66, 75 |
| `.env.example` placeholder only | Satisfied. Contains a clearly marked placeholder. | `backend/.env.example` |
| No usable credentials in current tracked files | Satisfied. Scan found only field names, test literals, documentation references and placeholder. | redacted scan of `git ls-files` |
| Git history | **Not clean.** Earlier commits still contain the pre-Phase-1 credential. No value is reproduced here. Rotation is required. | `git log -S` found history hits; current files clean |

### 4.2 Additional note

`scripts/verify.bat` now fails early if `PGPASSWORD` is unset. This is functionally correct and avoids embedding a secret, but the README setup section should explicitly mention `PGPASSWORD` for that script; currently it documents `POSTGRES_PASSWORD` only.

---

## Review 5 — Tests

### 5.1 Independent execution

I ran only the three new database-free test modules:

```bash
cd backend
.venv/Scripts/python.exe -m pytest -p no:cacheprovider -q \
  tests/test_dataset_api_contract.py \
  tests/test_config_security.py \
  tests/test_preprocessing_variants.py
```

Result: **9 passed, 0 failed, 0 skipped**. Working tree remained clean.

### 5.2 Test quality

| File | Assessment |
|---|---|
| `test_preprocessing_variants.py` | Tests real helper behavior for numeric/binary inverse round-trip, categorical rejection and deterministic subset selection. It does **not** test `build_preprocessed_variant` end-to-end, query-vector transformation, provenance payloads, failure cleanup, duplicate variants, `standardize_all` inverse, missing-class guard, or DB lifecycle. |
| `test_dataset_api_contract.py` | Direct function calls with monkeypatched DB helpers verify that the field is selected and returned. However, asserting `"preprocessing_mode" in captured["sql"]` is an implementation detail, and calling the endpoint function directly does not exercise FastAPI `response_model` validation. A `TestClient` test with dependency overrides would be stronger. |
| `test_config_security.py` | Good behavioral coverage for required/empty password, repr hiding and URL encoding. It does not test that `.env` loading works end-to-end or that no secret appears in logs. |

### 5.3 Tests requiring a real database

Not executed in this review: vector-store integration tests, runner integration tests, timing tests, dataset-statistics tests, artifact verifier, and any benchmark. These require a real PostgreSQL + pgvector database and in several cases create/delete synthetic datasets.

---

## Review 6 — Final decision and findings

| ID | Severity | Finding | Evidence | Required action |
|---|---|---|---|---|
| F1 | **BLOCKER** | Failed variant creation can leave an orphan `datasets` row and incomplete vector tables because insertion autocommits before table creation/COPY and there is no cleanup. | `run_experiments.py::_insert_variant_dataset`, `build_preprocessed_variant`; `core/database.py::get_pool` | Add transaction/compensating cleanup before any DB-writing smoke test. |
| F2 | HIGH | No end-to-end test for derived variants; highest-priority scientific fix is only unit-tested at helper level. | `test_preprocessing_variants.py` | Add isolated DB integration test with synthetic dataset. |
| F3 | HIGH | Variant provenance lacks parent fingerprint, exact selected ids and fitted-state hash. | `build_preprocessed_variant` return dict, lines 505–514 | Add to `config.extra` and `benchmark_orders.json`. |
| F4 | HIGH | Previous credential remains in Git history; rotation required. | history scan; current files clean | Rotate password outside repository. |
| F5 | MEDIUM | Repeated runs create duplicate variants; no idempotency/dedup. | `_insert_variant_dataset`; no uniqueness constraint | Decide explicit run policy before full benchmark. |
| F6 | MEDIUM | Legacy destructive helpers `subset_dataset`/`restore_pool` remain callable and can mutate an original table if invoked directly. | `run_experiments.py`, lines 308–398 | Remove, rename `legacy_`, or guard with explicit opt-in. |
| F7 | MEDIUM | API contract tests do not exercise FastAPI response validation. | `test_dataset_api_contract.py` | Add `TestClient` test. |
| F8 | MEDIUM | Authoritative thesis builder reads the live DB and overwrites thesis figures/docx; safe only after a new verified run. | `build_final_thesis.py`, lines 37, 171–172, 940–942 | Do not run until new benchmark is approved. |
| F9 | LOW | Unrelated internal memory files and prior audit report were committed with Phase 1. | commit `c92d620` | Keep branch history clean in future. |
| F10 | LOW | `verify.bat` docs mention `POSTGRES_PASSWORD` but not `PGPASSWORD`. | `README.md` setup section | Add one sentence. |
| F11 | LOW | Continuous column could be misclassified as binary in an unusually small subset. | `preprocessing.py::infer_column_types` | Document or add Covertype-specific guard. |

The preprocessing architecture itself is sound and does not require redesign: each pool gets its own dataset row, state, vector tables and fingerprint; original data is not mutated. The blocking issue is lifecycle safety, not the scientific design.

**Recommendation: B. Fix specific issues before proceeding.**
