# PostgreSQL-Based KNN Classification Platform

**Design and Evaluation of a PostgreSQL-Based K-Nearest Neighbors Classification Platform Using Exact and Approximate Vector Search**

A master's course project that implements an interactive platform for KNN
classification where neighbour retrieval happens **inside PostgreSQL** using
the [pgvector](https://github.com/pgvector/pgvector) extension, and compares
exhaustive (exact) search against an approximate HNSW index search.

> **Research question** — Can approximate nearest-neighbour search in
> PostgreSQL improve the performance of a KNN classification system while
> maintaining acceptable classification quality?

---

## Repository layout

```
ml-semester-project/
├── backend/          Python + FastAPI service (data, retrieval, evaluation)
│   ├── app/
│   │   ├── api/      HTTP endpoints (datasets, experiments, health)
│   │   ├── core/     Settings + connection pool
│   │   ├── db/       Schema definition and bootstrap
│   │   └── services/ The research logic
│   │       ├── classification.py   Majority voting + metrics (incl. tie-aware Recall@K)
│   │       ├── dataset_stats.py    Fingerprint, duplicate-vector and tie statistics
│   │       ├── datasets.py         Ingestion, splitting, Covertype loader
│   │       ├── experiments.py      Orchestration + persistence
│   │       ├── metrics.py          Distance metric definitions
│   │       ├── preprocessing.py    Leakage-free preprocessing
│   │       └── vector_store.py     Exact vs HNSW retrieval (pgvector)
│   ├── scripts/                    Batch runner + dataset statistics
│   └── tests/        pytest suite (164 tests)
├── frontend/         Next.js 16 + TypeScript + Tailwind v4 (bilingual EN/FA)
│   ├── app/[locale]/ Overview · Datasets · Experiments · Compare · Methodology
│   ├── components/   UI + feature components
│   ├── i18n/         next-intl routing
│   └── messages/     en.json · fa.json
├── docs/             Project documentation
├── pgvector/         Locally built pgvector 0.8.7 (see Setup)
└── scripts/          Root-level convenience scripts
```

## Architecture

| Layer | Technology | Responsibility |
| --- | --- | --- |
| Frontend | Next.js 16, TypeScript, Tailwind v4, Recharts, next-intl | Upload datasets, configure experiments, visualise results |
| Backend | Python 3.12, FastAPI, psycopg 3, scikit-learn | Preprocessing, retrieval orchestration, classification, evaluation |
| Database | PostgreSQL 18, pgvector 0.8.7 | Feature-vector storage, exact scan, HNSW index, result storage |

---

## Setup

### 1. PostgreSQL + pgvector

The platform needs PostgreSQL with the pgvector extension.

**pgvector on Windows (no admin rights required).** PostgreSQL 18 exposes two
GUCs that let a non-administrator install an extension from a writable
directory:

* `extension_control_path` — where `.control` / `.sql` files live
* `dynamic_library_path` — where the extension shared library lives

```powershell
# 1. Build pgvector from source against your PostgreSQL installation
cd pgvector-src
cmd /c "call `"C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat`" && nmake /F Makefile.win PGROOT=`"C:\Program Files\PostgreSQL\18`""

# 2. Install into a writable directory. Note the `extension` sub-directory:
#    PostgreSQL appends `/extension` to every non-$system entry of
#    extension_control_path.
$PGVECTOR = "<project>\pgvector"
mkdir "$PGVECTOR\lib", "$PGVECTOR\extension"
copy vector.dll                "$PGVECTOR\lib\"
copy vector.control            "$PGVECTOR\extension\"
copy sql\vector--0.8.7.sql     "$PGVECTOR\extension\"

# 3. Let the PostgreSQL service account read the files. The service runs as
#    NT AUTHORITY\NETWORK SERVICE, which cannot read files under a user
#    profile by default.
icacls "$PGVECTOR" /grant "NT AUTHORITY\NETWORK SERVICE:(OI)(CI)RX" /T

# 4. Point PostgreSQL at it (superuser)
psql -c "ALTER SYSTEM SET extension_control_path = '$PGVECTOR';"
psql -c "ALTER SYSTEM SET dynamic_library_path = '$PGVECTOR/lib';"
psql -c "SELECT pg_reload_conf();"
```

The backend creates the application database and enables the extension
automatically on startup.

> **Check:** `psql -c "SELECT extversion FROM pg_extension WHERE extname='vector'"`
> should print `0.8.7`.

### 2. Backend

```powershell
cd backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs: <http://127.0.0.1:8000/docs>

### 3. Frontend

```powershell
cd frontend
bun install
Copy-Item .env.example .env.local
bun run dev
```

Open <http://localhost:3000> — the interface is bilingual (English / فارسی)
with an RTL-aware Persian layout.

---

## How the experiment works

### Preprocessing (leakage-free)

1. A stratified train/test split is applied.
2. The scaler and the categorical mappings are **fitted on the training split
   only**, then applied unchanged to the test split.
3. In the primary `covertype` mode the 10 continuous features are standardised
   and the 44 binary indicators are kept as 0/1, because they are already
   dimensionless and standardising them would destroy their sparsity without
   changing their relative distances. Arbitrary uploaded CSVs use median
   imputation, one-hot encoding of nominal columns, and standardisation of the
   resulting numeric block. An explicit `standardize_all` mode is available
   for comparison; it is reported separately and is not the primary setup.
4. Vectors are written to PostgreSQL with `COPY` for throughput.

### Exact baseline

The exact search uses the pgvector **distance function**
(`l2_distance(embedding, $1)`), which the planner can never satisfy with an
index. This guarantees an exhaustive sequential scan and makes it a true
ground-truth baseline, independent of whether an HNSW index happens to exist.

```sql
SELECT id, label, l2_distance(embedding, $1) AS distance
FROM fv_1
ORDER BY l2_distance(embedding, $1)
LIMIT 5;
```

### Approximate search

The HNSW run uses the **operator** form (`embedding <-> $1`), which pgvector's
HNSW index can satisfy:

```sql
CREATE INDEX fv_1_embedding_hnsw_idx ON fv_1
  USING hnsw (embedding vector_l2_ops) WITH (m = 16, ef_construction = 64);

SET hnsw.ef_search = 100;
SELECT id, label, embedding <-> $1 AS distance
FROM fv_1
ORDER BY embedding <-> $1
LIMIT 5;
```

The planner's choice of access path is verified with `EXPLAIN (ANALYZE)` on
the first query of every batch and reported in the UI. A HNSW run whose plan
did not use the named index is never counted as an HNSW result; the batch
runner prints a warning when that happens.

### Repetitions and counts

`--repetitions N` executes **N complete timed passes** over the same query
set, in the same query order. It is not a multiplier applied to a counter.
Every individual latency is stored in `query_latencies` with its repetition
number, so a run with 1,000 queries and 3 repetitions stores exactly 3,000
rows. The warm-up query runs once, before the first timed query, and is never
stored.

Four quantities are reported separately:

| Quantity | Example |
| --- | --- |
| Held-out test pool | 3,000 rows |
| Unique evaluated queries | 1,000 |
| Repetitions | 3 |
| Latency observations | 3,000 |

Repetitions expose run-to-run variability. They are **not** independent
samples of a population: the query set, order and index are held fixed. The
per-repetition metrics are written to `experiment_repetitions`; cross-
repetition medians, standard deviations and 95 % intervals are exported
alongside the sweep.

### Measured metrics

| Category | Metric |
| --- | --- |
| Classification | Accuracy, macro precision / recall / F1, confusion matrix, per-class metrics |
| Retrieval | Conventional Recall@K against the deterministic exact neighbour ids |
| Retrieval | Tie-aware Recall@K against every id tied at the K-th distance (documented tolerance) |
| Latency | Average, median, p95, p99 per query, over every recorded observation |
| Index | Fresh build time, reuse flag, size, detected access path |

Conventional Recall@K counts an equally distant but differently identified
neighbour as a miss. Tie-aware Recall@K treats every id within
`boundary × (1 + 1e-6) + 1e-6` of the exact K-th distance as a valid answer.
Both are reported; neither replaces the other. On the Covertype pool used
here the two coincide because zero boundary ties were measured (see
[`docs/results/README.md`](docs/results/README.md)).

---

## Why one vector table per dataset

pgvector's HNSW index requires the indexed column to declare a **fixed
dimension** (`vector(54)`, not a bare `vector`). It also applies any residual
filter *after* the index scan, so a single shared table holding several
datasets would retrieve neighbours from the wrong dataset and silently
corrupt Recall@K.

Each dataset therefore owns a pair of tables:

* `fv_<dataset_id>` — the training pool (the HNSW index lives here)
* `tv_<dataset_id>` — the held-out query pool

Both use a dimensionally-typed `vector(dim)` column, which lets PostgreSQL
choose a clean HNSW index scan with no residual filter.

---

## Running the experiment matrix

```powershell
cd backend
.venv\Scripts\python -m scripts.run_experiments `
    --dataset-id 116 `
    --sizes 10000,50000 `
    --ef-search 10,25,50,100,200,400 `
    --k 5 --metric l2 `
    --max-queries 1000 `
    --repetitions 3 --seed 0 `
    --output ..\docs\results\sweep.csv `
    --json ..\docs\results\sweep.json `
    --repetition-details ..\docs\results\repetition_details.csv `
    --repetition-summary ..\docs\results\repetition_summary.csv
```

Measured output on dataset 116 (1,000 queries, 3 repetitions, L2, `k = 5`):

```
    size   method   k    ef reps   recall      acc  macroF1     p50ms     p99ms   buildms   idx       x
-------------------------------------------------------------------------------------------------------
    9999    exact   5     -    3   1.0000   0.7750   0.5069     2.769     3.728         - False       -
    9999     hnsw   5    10    3   0.9952   0.7760   0.5070     0.211     0.331     597.6  True    13.1
    9999     hnsw   5    25    3   0.9996   0.7750   0.5069     0.246     0.383         -  True    11.3
    9999     hnsw   5    50    3   0.9998   0.7750   0.5069     0.297     0.443         -  True     9.3
    9999     hnsw   5   100    3   1.0000   0.7750   0.5069     0.387     0.555         -  True     7.2
    9999     hnsw   5   200    3   1.0000   0.7750   0.5069     0.564     0.786         -  True     4.9
    9999     hnsw   5   400    3   1.0000   0.7750   0.5069     0.908     1.266         -  True     3.0
   50000    exact   5     -    3   1.0000   0.8560   0.7314    11.226    12.596         - False       -
   50000     hnsw   5    10    3   0.9822   0.8550   0.7309     0.230     0.358    3239.0  True    48.8
   50000     hnsw   5    25    3   0.9926   0.8580   0.7403     0.269     0.413         -  True    41.7
   50000     hnsw   5    50    3   0.9978   0.8570   0.7400     0.331     0.490         -  True    33.9
   50000     hnsw   5   100    3   0.9998   0.8570   0.7400     0.442     0.763         -  True    25.4
   50000     hnsw   5   200    3   0.9998   0.8570   0.7400     0.655     0.993         -  True    17.1
   50000     hnsw   5   400    3   0.9998   0.8570   0.7400     1.079     1.528         -  True    10.4
```

`x` is the exact/HNSW median-latency ratio. The HNSW index is built once per
pool size and **reused** across `ef_search` values; those rows record
`index_build_ms = NULL`, `index_reused = True`, and the original build time in
`index_reused_build_ms`.

Dataset-level duplicate and tie statistics, used to interpret Recall@K:

```powershell
.venv\Scripts\python -m scripts.dataset_statistics `
    --dataset-id 116 --k 5 --metric l2 --max-queries 1000 `
    --output ..\docs\results\duplicate_stats.json
```

---

## Tests

```powershell
cd backend
.venv\Scripts\python -m pytest
```

The suite contains **164 tests** and covers the classification/metric maths
(including conventional and tie-aware Recall@K), the leakage contract of the
preprocessing pipeline, the stratified splitting, and — against a live
database — the exact/HNSW retrieval behaviour, index build/reuse
bookkeeping, repetition accounting, persisted latency-row counts, tied
boundary detection, and the batch runner's CSV/JSON consistency.

---

## Further reading

| Document | Contents |
| --- | --- |
| [`docs/Thesis-KNN-PostgreSQL-Platform.docx`](docs/Thesis-KNN-PostgreSQL-Platform.docx) | The full thesis — problem statement, related work, methodology, implementation, results and discussion |
| [`docs/Project-Guide.docx`](docs/Project-Guide.docx) | The function-by-function tour of the database, the backend and the frontend |
| [`docs/architecture.md`](docs/architecture.md) | The design decisions and the traps they avoid |
| [`docs/api.md`](docs/api.md) | The HTTP API reference |
| [`docs/experiment-plan.md`](docs/experiment-plan.md) | The experimental design and the threats to validity |
| [`docs/results/README.md`](docs/results/README.md) | The measured results and their interpretation |

Both Word documents are generated from source by the scripts in
`docs/tools/`, so they can be regenerated whenever the project changes:

```powershell
cd backend
.venv\Scripts\python ..\docs\tools\build_thesis.py
.venv\Scripts\python ..\docs\tools\build_guide.py
.venv\Scripts\python ..\docs\tools\verify_snippets.py ..\docs\Project-Guide.docx
```

`verify_snippets.py` checks that every Python and TypeScript listing in the
documents still matches the source tree, and reports any divergence.
