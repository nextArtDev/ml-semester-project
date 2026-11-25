# Reproducibility Guide

**Audited base commit:** `49a5abab0507aa06d472f1d64dcd787c4e65e07c`  
**Branch:** `phase3-benchmark-design`  
**Scope:** reproduce the validated Phase 3 benchmark, validation, figures, tables and thesis.

Secrets are never included here. Set passwords locally and keep them out of tracked files.

---

## 1. Required software

| Component | Version used |
|---|---|
| Python | 3.12.1 |
| PostgreSQL | 18.0 |
| pgvector | 0.8.7 |
| NumPy | 2.5.4 |
| pandas | 3.1.1 |
| matplotlib | 3.11.2 |
| python-docx | installed via `docs/tools/requirements-doc.txt` |

Windows paths below use Git Bash. Substitute equivalent paths on other systems.

## 2. Clone and inspect

```bash
cd "C:/Users/aria/Desktop/NextJS Projects/ml-semester-project"
git status --short --branch
git rev-parse HEAD
```

Expected branch at audit time: `phase3-benchmark-design`. Expected audited commit: `49a5aba`.

## 3. Python environment

```bash
cd backend
python -m venv .venv
.venv/Scripts/python.exe -m pip install --upgrade pip
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pip install -r ../docs/tools/requirements-doc.txt
```

## 4. Database and credentials

Create a local database user/database, install pgvector 0.8.7, then copy the template and edit it locally:

```bash
cd backend
cp .env.example .env
# Edit .env with a local editor and set POSTGRES_USER, POSTGRES_PASSWORD,
# POSTGRES_HOST, POSTGRES_PORT and POSTGRES_DB. Do not commit .env.
```

Required environment variables:

| Variable | Purpose |
|---|---|
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB` | backend database connection |
| `ALLOW_PHASE1_DB_TESTS`, `PHASE1_TEST_DATABASE_URL` | enable guarded Phase 1 lifecycle tests |
| `ALLOW_PHASE2_DB_TESTS`, `PHASE2_TEST_DATABASE_URL` | enable guarded Phase 2 remediation tests |
| `PGPASSWORD` | required only by `scripts/verify.bat`; set in the shell, not in tracked files |

Initialize the schema:

```bash
cd backend
.venv/Scripts/python.exe -c "from app.db.init_db import init_schema; init_schema(); print('schema OK')"
```

See `README.md` for the Windows pgvector build/copy steps. A historical credential remains in old Git history; rotate that credential in PostgreSQL and do not rewrite historical commits without an explicit decision.

## 5. Dataset preparation

The validated experiments use parent dataset 116: 50,000 training rows and 3,000 held-out rows, parent fingerprint `6baa8fa283ea5a8267d3e964eedf5d2a`. On the audited database this dataset is already present.

On a fresh database, start the API and ingest the Covertype source:

```bash
cd backend
.venv/Scripts/python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In a second shell:

```bash
# API_PREFIX in .env.example is /api
curl -X POST "http://127.0.0.1:8000/api/datasets/covertype" \
  -H "Content-Type: application/json" \
  -d '{"name":"Covertype 50k","max_rows":581012,"test_size":3000,"max_train_rows":50000,"seed":42,"preprocessing_mode":"covertype"}'
```

Record the returned `dataset_id`. Reproducing the exact historical experiment uses dataset id 116; a fresh ingestion may receive a different id, so substitute that id in subsequent commands and verify the fingerprint/preprocessing state rather than assuming identity.

## 6. Derived datasets and indexes

The runner creates the 10,000-row and 50,000-row variants and builds the indexes automatically. Parameters are fixed by the approved protocol:

- `m=16`
- `ef_construction=64`
- `ef_search=10,50,100,400`
- `k=5`
- L2 distance

No manual index build is required. `ensure_hnsw_index()` verifies the catalog and reuses a valid index.

## 7. Run the approved natural benchmark

```bash
cd backend
RUN_ID="phase3-natural-v1-<your-unique-suffix>"
mkdir -p ../docs/results/phase3

.venv/Scripts/python.exe -m scripts.run_experiments \
  --dataset-id 116 --sizes 10000,50000 \
  --planner-mode natural --ef-search 10,50,100,400 \
  --k 5 --metric l2 --max-queries 0 \
  --repetitions 3 --rounds 3 --order-seed 0 --seed 0 \
  --query-sampling stratified --query-sample-seed 0 \
  --hnsw-m 16 --hnsw-ef-construction 64 \
  --run-id "$RUN_ID" \
  --output ../docs/results/phase3/sweep.csv \
  --json ../docs/results/phase3/sweep.json \
  --repetition-details ../docs/results/phase3/repetition_details.csv \
  --repetition-summary ../docs/results/phase3/repetition_summary.csv \
  --query-samples ../docs/results/phase3/query_samples.json \
  --benchmark-orders ../docs/results/phase3/benchmark_orders.json \
  --variant-ids ../docs/results/phase3/variant_ids.json
```

The 10k/ef=400 natural cell may select a sequential scan. Under the approved option A it must be recorded as failed, not converted to a successful HNSW measurement.

## 8. Run controlled HNSW diagnostics separately

Use the protocol's controlled diagnostic command in `PHASE3-BENCHMARK-PROTOCOL.md`, with a distinct run id. Never combine controlled rows with natural rows in one performance group.

## 9. Validate raw artifacts and recalculate metrics

```bash
cd backend
.venv/Scripts/python.exe scripts/validate_phase3_results.py
```

This writes `docs/results/validated/summary.csv` and `statistical_summary.json`. It independently recomputes Recall@5, classification metrics, latency statistics, provenance hashes and access-path validity from raw artifacts/database records.

The legacy verifier remains available but is not compatible with the approved four-ef matrix or expected failed HNSW rows:

```bash
.venv/Scripts/python.exe -m scripts.verify_results \
  --results-dir ../docs/results/phase3 \
  --run-id "$RUN_ID" --expected-runs 30 --expected-rounds 3 \
  --expected-queries 3000 --expected-repetitions 3
```

## 10. Regenerate figures and tables

```bash
cd backend
.venv/Scripts/python.exe ../scripts/generate_research_figures.py
```

Outputs: `docs/results/figures/figure_01...figure_08` and six CSVs in `docs/results/tables/`. The Phase 4 audit confirmed byte-identical regeneration.

## 11. Rebuild the thesis

```bash
cd backend
.venv/Scripts/python.exe ../docs/thesis/build_thesis_docx.py
```

This reads `docs/thesis/THESIS.md` and copied validated figures/tables, and writes `docs/thesis/THESIS.docx`. Refresh the Word TOC field after opening.

## 12. Tests

Use fresh isolated databases:

```bash
cd backend
export POSTGRES_DB=test_phase3_audit
.venv/Scripts/python.exe -c "from app.db.init_db import init_schema; init_schema()"
export ALLOW_PHASE1_DB_TESTS=1 ALLOW_PHASE2_DB_TESTS=1
export PHASE1_TEST_DATABASE_URL="<local-test-url-1>"
export PHASE2_TEST_DATABASE_URL="<local-test-url-2>"
.venv/Scripts/python.exe -m pytest -p no:cacheprovider tests
```

Audited result on fresh databases: **272 passed, 1 skipped**. Reusing a dirty test database can produce duplicate dataset-id errors; recreate the database before a clean run.
