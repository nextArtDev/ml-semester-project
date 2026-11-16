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
│   │       ├── classification.py   Majority voting + metrics
│   │       ├── datasets.py         Ingestion, splitting, Covertype loader
│   │       ├── experiments.py      Orchestration + persistence
│   │       ├── metrics.py          Distance metric definitions
│   │       ├── preprocessing.py    Leakage-free preprocessing
│   │       └── vector_store.py     Exact vs HNSW retrieval (pgvector)
│   ├── scripts/run_experiments.py  Batch experiment matrix runner
│   └── tests/        pytest suite (47 tests)
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
3. Vectors are written to PostgreSQL with `COPY` for throughput.

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
every HNSW run and reported in the UI.

### Measured metrics

| Category | Metric |
| --- | --- |
| Classification | Accuracy, macro precision / recall / F1, confusion matrix, per-class metrics |
| Retrieval | Recall@K against the exact neighbour set |
| Latency | Average, median, p95, p99 per query |
| Index | HNSW construction time, detected access path |

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
    --dataset-id 1 `
    --sizes 5000,15000,581000 `
    --ef-search 10,25,50,100,200,400 `
    --k 5 `
    --max-queries 2000 `
    --output ..\docs\results\sweep.csv
```

Sample output:

```
    size   method   k    ef   recall      acc  macroF1     p50ms     p95ms   buildms
------------------------------------------------------------------------------------
    5000    exact   5     —   1.0000   0.6967   0.3995     1.272     1.428         —
   15000    exact   5     —   1.0000   0.7667   0.4853     3.435     3.804         —
    5000     hnsw   5    25   0.9340   0.6700   0.3726     0.253     0.353    655.2
    5000     hnsw   5   100   0.9660   0.6900   0.3888     0.349     0.443    655.2
   15000     hnsw   5    25   0.9007   0.7533   0.4763     0.258     0.343   1942.5
   15000     hnsw   5   100   0.9467   0.7600   0.4858     0.376     0.489   1942.5
```

The HNSW index is **reused** across runs that share the same metric and build
parameters, so an `ef_search` sweep does not pay the construction cost again.

---

## Tests

```powershell
cd backend
.venv\Scripts\python -m pytest
```

The suite covers the classification/metric maths, the leakage contract of the
preprocessing pipeline, the stratified splitting, and — against a live
database — the exact/HNSW retrieval behaviour, index reuse, and the
monotonicity of Recall@K as `ef_search` grows.

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
