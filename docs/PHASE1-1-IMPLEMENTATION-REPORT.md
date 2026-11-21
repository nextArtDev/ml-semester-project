# Phase 1.1 — Derived Dataset Lifecycle and Reproducibility Hardening: Implementation Report

**Branch:** `phase1-stabilization`  
**Date:** 2026-11-21  
**Commit created:** no  
**Benchmark executed:** no  
**Thesis builder executed:** no  
**Normal experiment database modified:** no  
**Isolated test database used:** `test_phase11`  

---

## 1. Changed files

| File | Status | Purpose |
|---|---|---|
| `backend/app/core/database.py` | modified | Added `transaction()` helper that borrows one pooled connection, disables autocommit, commits/rolls back, and restores autocommit. |
| `backend/app/services/reproducibility.py` | new | Canonical JSON and SHA-256 helpers for selected ids, preprocessing state and complete variant identity. |
| `backend/app/services/preprocessing.py` | modified | Added `preprocessor_from_state()` so a stored state can be replayed without refitting. |
| `backend/scripts/run_experiments.py` | modified | Atomic variant creation, transaction-local DDL/COPY helpers, extended provenance, `--variant-ids` artifact, immutable-variant policy, guarded legacy helpers. |
| `backend/tests/test_database_transaction.py` | new | Unit tests for commit, rollback and autocommit restoration. |
| `backend/tests/test_preprocessing_variants.py` | modified | Added provenance/hash tests and legacy-guard tests. |
| `backend/tests/test_variant_lifecycle_integration.py` | new | Isolated-DB end-to-end lifecycle and failure-injection tests. Explicitly guarded. |
| `backend/tests/test_dataset_api_contract.py` | modified | Added FastAPI `TestClient` tests exercising response-model validation and 404 behavior. |
| `backend/tests/test_training_subsample.py` | modified | Existing destructive helper calls now pass `allow_destructive=True` explicitly. |
| `README.md` | modified | Documented atomic lifecycle, immutable variants, provenance algorithm, `--variant-ids`, credential rotation and `POSTGRES_PASSWORD` vs `PGPASSWORD`. |
| `docs/architecture.md` | modified | Documented transaction design and variant identity policy. |
| `docs/PHASE1-REVIEW-REPORT.md` | pre-existing untracked | Review report from previous task; left unchanged. |
| `docs/PHASE1-1-IMPLEMENTATION-REPORT.md` | new | This report. |
| `.workbuddy-ai/memory/2026-11-21.md` | modified | Internal work log; pre-existing modification continued for this task. |

No frontend file changed. No result CSV/JSON artifact changed. No existing thesis document changed.

---

## 2. Findings and implementation details

### A — BLOCKER: non-atomic variant creation

**Root cause.** `build_preprocessed_variant()` used separate autocommit connections for `INSERT INTO datasets`, `create_dataset_tables()` and two `insert_vectors()` calls. A failure after the first statement could leave an orphan dataset row or an incomplete table set.

**Implementation.** Added `database.transaction()`:

- borrows one pooled connection;
- records the original `conn.autocommit`;
- sets `conn.autocommit = False`;
- yields that one connection to the caller;
- commits only after the body succeeds;
- rolls back on any exception, including commit failure;
- restores the original autocommit setting in `finally`.

`build_preprocessed_variant()` now performs all four resource-creating operations with one cursor from that one connection:

1. `_insert_variant_dataset_row(cur, ...)`
2. `_create_variant_tables(cur, ...)`
3. `_copy_variant_vectors(cur, ..., "train", ...)`
4. `_copy_variant_vectors(cur, ..., "test", ...)`

`dataset_fingerprint(variant_id)` is computed after commit, when the variant is fully visible on a normal pooled connection. Failure at that point cannot leave a partial variant.

**Design choice.** A single real transaction was practical, so no compensating delete/drop path was added. Compensating cleanup is more error-prone and would have separate failure modes.

### B — Provenance and reproducibility

**Root cause.** Earlier provenance had a parent id, derived id, size, seed, mode, state source and derived fingerprint, but not the parent fingerprint, exact selected-id hash/count, fitted-state hash or a complete identity hash.

**Implementation.** New `app.services.reproducibility` defines:

- `canonical_json(value)` — `json.dumps(..., sort_keys=True, separators=(",", ":"), ensure_ascii=False)`;
- `sha256_text(text)` — SHA-256 over UTF-8 bytes;
- `selected_ids_provenance(ids)` — sorted integer ids plus count and hash;
- `preprocessing_state_hash(state)` — hash of `PreprocessState.to_dict()`;
- `variant_identity_hash(provenance)` — hash of parent fingerprint, target size, seed, selected-id count/hash, preprocessing mode and fitted-state hash.

`build_preprocessed_variant()` now returns `(metadata, selected_ids)`. Exact ids are not stored in `config.extra.pool_variant`. They are exported once per variant with the new `--variant-ids <path>` option.

`benchmark_orders.json` now includes parent fingerprint, derived fingerprint, selected-id count/hash, preprocessing-state hash, variant identity hash and state source in every round entry.

### C — End-to-end lifecycle tests

**Implementation.** `backend/tests/test_variant_lifecycle_integration.py` covers:

1. successful small variant creation;
2. refit on the selected training rows;
3. query vectors transformed with the derived state;
4. parent and derived fingerprints;
5. selected-id and preprocessing-state hashes;
6. provenance consistency in experiment config and benchmark order;
7. failure immediately after dataset insertion;
8. failure after creation of the first vector table;
9. failure during vector insertion;
10. parent dataset unchanged after each failure;
11. no orphan derived datasets or partial tables after each failure;
12. successful variant consumed by `run_experiment`.

The tests are guarded by `ALLOW_PHASE1_DB_TESTS=1` and `PHASE1_TEST_DATABASE_URL`. The fixture requires `current_database()` to differ from `settings.postgres_db` and to start with `test_`. It applies schema statements and cleans up only rows/tables with the `pytest-phase11-` prefix.

These tests were executed against an isolated local database named `test_phase11`.
The normal experiment database was not touched. Initial failures were caused by
test-harness assumptions (absolute derived-dataset counts and exact float-string
comparison); they were corrected, and all six lifecycle tests now pass.

### D — Duplicate variant policy

**Chosen policy:** derived variants are **immutable retained artifacts**.

Every size-sweep invocation creates new derived dataset rows and vector tables. No automatic reuse, deletion or deduplication is performed. Reuse would require matching the complete variant identity hash, not merely target size and seed.

Consequence: database growth is intentional and predictable. Each requested pool size adds one dataset row, two vector tables and any experiments that consume them. Cleanup is a manual decision after archival, never an implicit runner action.

### E — Legacy destructive helpers

`restore_pool()` and `subset_dataset()` remain for existing tests but now require explicit `allow_destructive=True`. Their implementations were renamed `legacy_restore_pool()` and `legacy_subset_dataset()`. The normal benchmark path does not call them.

Existing calls in `test_training_subsample.py` were updated to opt in explicitly. New unit tests assert that calling either wrapper without the flag raises `RuntimeError`.

### F — API response testing

`test_dataset_api_contract.py` now uses FastAPI `TestClient` to exercise `/api/datasets` and `/api/datasets/{id}` through actual route handling and response-model validation. Lifespan hooks are neutralised so the test does not touch the database. It checks `preprocessing_mode` and the 404 path. The earlier monkeypatch-based unit tests remain.

### G — Credentials and Covertype-specific type guard

**Credentials.** README now states:

- `POSTGRES_PASSWORD` is the backend's required environment value;
- `PGPASSWORD` is read by `scripts/verify.bat` from the parent shell;
- neither belongs in tracked files;
- the previously exposed password must be rotated in PostgreSQL manually, without rewriting Git history.

**Column-type guard.** `build_preprocessed_variant()` now compares the refitted `numeric_columns`, `binary_columns` and `categorical_columns` with the parent state. If a small subsample would silently change a known column's type classification, creation stops before any database write.

---

## 3. Transaction and cleanup design

| Failure point | Behavior |
|---|---|
| Before `INSERT INTO datasets` | Nothing was written. No cleanup needed. |
| Immediately after dataset insertion | Exception exits `transaction()`, rollback removes the dataset row. |
| After only `fv_<id>` is created | Rollback removes the dataset row and the table/index. |
| During train COPY | Rollback removes partial rows, both tables and the dataset row. |
| During test COPY | Train rows, test rows, tables and dataset row all roll back. |
| After successful commit, during fingerprint read | Variant is complete; no partial state exists. Fingerprint may be absent from artifacts, but the variant itself is valid. |
| Any failure | Original `autocommit=True` is restored on the borrowed pooled connection. |

The connection and cursor share the transaction because `transaction()` yields one `psycopg.Connection`, and all helpers receive a cursor created from that same connection. PostgreSQL treats DDL and COPY inside a transaction atomically.

---

## 4. Provenance format

Example with placeholder hashes only:

```json
{
  "parent_dataset_id": 116,
  "dataset_id": 214,
  "target_size": 10000,
  "actual_size": 10000,
  "training_pool_seed": 0,
  "parent_fingerprint": "<64 hex chars>",
  "derived_fingerprint": "<64 hex chars>",
  "preprocessing_mode": "covertype",
  "selected_id_count": 10000,
  "selected_id_sha256": "<64 hex chars>",
  "preprocessing_state_sha256": "<64 hex chars>",
  "variant_identity_sha256": "<64 hex chars>",
  "state_source": "refit_on_variant_training_pool",
  "variant_policy": "immutable_retained"
}
```

Algorithms:

- canonical JSON: sorted keys, compact separators, UTF-8, non-ASCII preserved;
- selected ids: sorted ascending integers before JSON serialisation;
- hash: SHA-256, lowercase hex digest;
- preprocessing state: complete `PreprocessState.to_dict()`;
- identity: canonical JSON of parent fingerprint, target size, seed, selected-id count/hash, preprocessing mode and state hash.

The exact id array is written once to the `--variant-ids` artifact. To verify it, sort the ids, serialise canonically, compute SHA-256 and compare with `selected_id_sha256`.

---

## 5. Test coverage and exact results

Command executed:

```bash
cd backend
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
  tests/test_preprocessing_variants.py \
  tests/test_database_transaction.py \
  tests/test_variant_lifecycle_integration.py
```

Result:

```text
161 passed, 0 skipped, 0 failed
```

The run included the six isolated-DB lifecycle tests. One non-failing
Starlette/httpx deprecation warning was emitted by `TestClient`.

Also executed:

- Python AST syntax validation for all changed Python files: passed.
- `git diff --check`: passed.
- Frontend type checking was not run because no frontend file changed.

### `git status --short` at validation time

```text
 M .workbuddy-ai/memory/2026-11-21.md
 M README.md
 M backend/app/core/database.py
 M backend/app/services/preprocessing.py
 M backend/scripts/run_experiments.py
 M backend/tests/test_dataset_api_contract.py
 M backend/tests/test_preprocessing_variants.py
 M backend/tests/test_training_subsample.py
 M docs/architecture.md
?? backend/app/services/reproducibility.py
?? backend/tests/test_database_transaction.py
?? backend/tests/test_variant_lifecycle_integration.py
?? docs/PHASE1-REVIEW-REPORT.md
```

This report was created after that status capture. No commit was created.

### `git diff --stat` at validation time

```text
 .workbuddy-ai/memory/2026-11-21.md           |   3 +
 README.md                                    |  41 +++-
 backend/app/core/database.py                 |  28 +++
 backend/app/services/preprocessing.py        |  28 +++
 backend/scripts/run_experiments.py           | 304 +++++++++++++++++++++------
 backend/tests/test_dataset_api_contract.py   |  30 +++
 backend/tests/test_preprocessing_variants.py |  51 +++++
 backend/tests/test_training_subsample.py     |  14 +-
 docs/architecture.md                         |  13 ++
 9 files changed, 430 insertions(+), 82 deletions(-)
```

Untracked new files are not included in `git diff --stat`.

---

## 6. Database safety statement

- No SQL was executed against the normal experiment database during this task.
- An isolated `test_phase11` database was created on the local PostgreSQL server and used only by the guarded lifecycle tests; its `pytest-phase11-` rows/tables were cleaned by the fixture.
- No migration was created or run.
- No benchmark was executed.
- No existing dataset, vector table, experiment row or result artifact was modified.
- The thesis builder was not executed.
- The FastAPI `TestClient` test neutralises lifespan hooks and does not connect to the database.
- The new transaction helper is exercised only by a fake-connection unit test.
- The isolated-DB integration tests were written but not run because no isolated database was supplied.

---

## 7. Tests not executed and why

| Test / action | Reason |
|---|---|
| Existing DB integration tests | They would connect to the normal local experiment database and create/delete synthetic datasets. |
| `scripts.verify_results` | Reads the live experiment database. |
| Full benchmark | Explicitly forbidden in this phase. |
| Thesis builder | Explicitly forbidden; also writes thesis figures/docx. |
| Frontend typecheck | No frontend file changed. |

---

## 8. Remaining risks and limitations

1. The end-to-end rollback design is now verified on an isolated PostgreSQL database, including failures after insert, after one table and during COPY.
2. Derived variants are immutable and never deduplicated; database growth must be planned.
3. Inverse reconstruction depends on the existing `%.8g` vector-text precision. It is adequate for numeric/binary Covertype but is not a general raw-data archive.
4. Categorical one-hot datasets are rejected for size variants.
5. The old credential remains in Git history. Current tracked files are clean, but rotation is still required.
6. Legacy destructive helpers still exist, though they now require an explicit opt-in flag.
7. `verify_results.py` still checks artifact integrity, not independent scientific recomputation.
8. No new benchmark numbers were produced; previous 10k/50k cross-pool conclusions remain pending rerun.

---

## 9. Exact manual actions still required

1. Rotate the previously exposed PostgreSQL password in the database and any deployment environment. Do not rewrite Git history.
2. Back up the normal experiment database before any smoke benchmark.
3. Approve a small smoke benchmark with `--variant-ids` and inspect the generated provenance artifact.
4. Only after smoke validation, approve the full 10k/50k rerun under a new run id.

---

## 10. Recommendation

**A. Safe to prepare database backup and smoke-test planning.**
