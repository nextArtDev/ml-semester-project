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
│   │       ├── benchmark_order.py  Seeded balanced benchmark rounds
│   │       ├── classification.py   Majority voting + metrics (incl. tie-aware Recall@K)
│   │       ├── dataset_stats.py    Fingerprint, duplicate-vector and tie statistics
│   │       ├── datasets.py         Ingestion, splitting, Covertype loader
│   │       ├── experiments.py      Orchestration + persistence + phase timing
│   │       ├── metrics.py          Distance metric definitions
│   │       ├── preprocessing.py    Leakage-free preprocessing
│   │       └── vector_store.py     Exact vs HNSW retrieval (pgvector)
│   ├── scripts/                    Batch runner + dataset statistics
│   └── tests/        pytest suite (241 passed, 1 skip)
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
# Set POSTGRES_PASSWORD in backend/.env to your local PostgreSQL password.
# The backend refuses to start if this environment variable is missing or empty.
.venv\Scripts\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Interactive API docs: <http://127.0.0.1:8000/docs>

`POSTGRES_PASSWORD` is read by the backend from `backend/.env` (or the process
environment). `scripts/verify.bat` is a separate Windows shell script and reads
`PGPASSWORD` from the parent shell instead; set that variable in your terminal
before running it. Do not place either value in tracked files.

A password from an earlier version of this project is still present in Git
history. The password must be rotated in PostgreSQL outside this repository;
historical commits are intentionally not rewritten.

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

For a training-size sweep, each requested size is materialised as a derived
dataset with its own preprocessing state fitted on that size's training rows.
Registration, table creation and both COPY operations share one PostgreSQL
transaction, so a failed creation leaves no derived dataset or vector table.
The held-out queries are transformed with that same state. The original dataset
and its vector tables are not modified. This applies to numeric/binary datasets
such as Covertype; datasets with categorical one-hot blocks are rejected for
size variants because raw category values are not stored and cannot be
reconstructed safely. If a small sample changes a known continuous column's
type classification, variant creation also stops instead of silently changing
the feature geometry.

Derived variants are **immutable retained artifacts**. A new benchmark run
creates new derived dataset rows; nothing is deleted or deduplicated
automatically. Plan for database growth: each size adds one dataset row and two
vector tables, plus any experiments that consume them.

Provenance is compact. Every experiment's `config.extra.pool_variant` and every
`benchmark_orders.json` entry record the parent fingerprint, selected-ID count
and SHA-256, preprocessing-state SHA-256, variant identity hash, both dataset
ids and both fingerprints. Exact selected ids are exported once per variant
with `--variant-ids <path>`. Hashes use SHA-256 over canonical JSON
(`sort_keys=True`, separators `,` and `:`, UTF-8, non-ASCII preserved); selected
ids are sorted ascending before hashing. To verify an artifact, recompute the
hash from its id array and compare it with `selected_id_sha256`.

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

### Repetitions, rounds and counts

`--repetitions N` executes **N complete timed passes** over the same query set,
in the same query order. It is not a multiplier applied to a counter.
Every individual latency is stored in `query_latencies` with its repetition
number. The warm-up query runs once, before the first timed query, and is never
stored.

`--rounds R` runs the whole configuration matrix R times in a seeded order.
Each round contains every configuration exactly once. For the supported 7×3
matrix, exhaustive tests demonstrate that every configuration visits three
distinct positions and has both early and late slots; no perfect balance is
claimed for arbitrary matrix sizes. The order is derived from `--order-seed`,
persisted as `benchmark_round`, `benchmark_position`, `benchmark_seed` and
`benchmark_order`, and exported to `benchmark_orders.json`. The HNSW index is
built once per pool and reused across `ef_search` values and rounds.

Three distinct levels must not be conflated:

| Level | What varies | What it measures |
| --- | --- | --- |
| Query observations | Query id | Per-query latency over the whole held-out pool |
| Repetitions (3) | Nothing but time | Run-to-run timing variability |
| Benchmark rounds (3) | Configuration order only | Fairness of early/late position |

Five quantities are reported separately:

| Quantity | Final benchmark |
| --- | --- |
| Held-out test pool | 3,000 rows |
| Unique evaluated queries | 3,000 (all rows) |
| Repetitions | 3 |
| Benchmark rounds | 3 |
| Latency observations | 3,000 × 3 × 3 = 27,000 per configuration; 378,000 total |

None of these levels is an independent statistical sample of a population.
Per-repetition metrics are written to `experiment_repetitions`; cross-repetition
medians, standard deviations and 95 % intervals are exported alongside the
sweep.

### Query selection

The default runner mode is a seeded **stratified** sample: class slots are
allocated by largest remainder and rows are sampled within each class with a
seed derived from `--query-sample-seed` and the class label. The selected ids
are identical for the exact and every HNSW run, are persisted as
`evaluated_query_ids`, and are exported to `query_samples.json` together with
the pool and evaluated class distributions. The legacy `first` mode remains
available for reproducing older runs. Use `--max-queries 0` to evaluate the
whole held-out pool.

### Measured metrics

| Category | Metric |
| --- | --- |
| Classification | Accuracy, macro precision / recall / F1, confusion matrix, per-class metrics |
| Retrieval | Conventional Recall@K against the deterministic exact neighbour ids |
| Retrieval | Tie-aware Recall@K against every id tied at the K-th distance (documented tolerance) |
| Latency | Average, median, p95, p99 per query, over every recorded observation |
| Index | Fresh build time, reuse flag, size, detected access path |
| Timing | Total runtime and separate index preparation, ground truth, retrieval, metrics, persistence and orchestration intervals |

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
    --max-queries 0 `
    --repetitions 3 --seed 0 `
    --rounds 3 --order-seed 0 `
    --query-sampling stratified --query-sample-seed 0 `
    --run-id final-v3-full-pool `
    --output ..\docs\results\sweep.csv `
    --json ..\docs\results\sweep.json `
    --repetition-details ..\docs\results\repetition_details.csv `
    --repetition-summary ..\docs\results\repetition_summary.csv `
    --query-samples ..\docs\results\query_samples.json `
    --benchmark-orders ..\docs\results\benchmark_orders.json `
    --variant-ids ..\docs\results\variant_ids.json
```

Aggregated over the three rounds, the measured output on dataset 116 with all
3,000 held-out queries was:

```
   rows method  ef  recall    acc  macroF1    p50(ms)  speed-up
  10000  exact   -  1.0000  0.7553   0.5623      2.823         -
  10000   hnsw  10  0.9941  0.7547   0.5602      0.223      12.7x
  10000   hnsw  25  0.9989  0.7553   0.5613      0.257      11.0x
  10000   hnsw  50  0.9997  0.7553   0.5613      0.320       8.8x
  10000   hnsw 100  0.9999  0.7553   0.5623      0.416       6.8x
  10000   hnsw 200  0.9999  0.7553   0.5623      0.600       4.7x
  10000   hnsw 400  0.9999  0.7553   0.5623      1.004       2.8x
  50000  exact   -  1.0000  0.8440   0.7500     11.063         -
  50000   hnsw  10  0.9854  0.8423   0.7470      0.246      45.1x
  50000   hnsw  25  0.9942  0.8443   0.7499      0.280      39.6x
  50000   hnsw  50  0.9974  0.8437   0.7499      0.363      30.6x
  50000   hnsw 100  0.9989  0.8440   0.7500      0.465      23.8x
  50000   hnsw 200  0.9997  0.8440   0.7500      0.753      14.7x
  50000   hnsw 400  0.9998  0.8440   0.7500      1.224       9.1x
```

These figures evaluate the whole held-out pool; they differ from the preserved
[`v1/`](docs/results/v1/README.md) and [`v2/`](docs/results/v2/README.md)
artifacts and must not be mixed with them. The HNSW index is built once per
pool size and **reused** across all ef values and rounds.

Check artifact integrity with:

```powershell
.venv\Scripts\python -m scripts.verify_results `
    --run-id final-v3-full-pool --expected-runs 42 --expected-rounds 3 `
    --expected-queries 3000 --expected-repetitions 3
```

This checks artifact structure, counts, seeds, experiment ids, plan flags and
CSV/JSON/database agreement. It is an **artifact integrity check**, not an
independent scientific recomputation of recall, accuracy or latency from raw
per-query evidence. The raw per-query rows remain in the database tables
`query_latencies` and `experiment_repetitions`.

### Timing fields

`api_overhead_s` was removed because it charged ground-truth computation, index
preparation, metrics and persistence to "API overhead". The replacement fields,
all measured with `time.perf_counter()`, are:

| Field | Definition |
| --- | --- |
| `total_experiment_runtime_s` | Start of `run_experiment` through persistence |
| `index_preparation_s` | Build-or-reuse decision, `ANALYZE`, metadata readback |
| `index_build_s` | Fresh `CREATE INDEX` only; subset of preparation, 0 on reuse |
| `ground_truth_computation_s` | Exhaustive neighbours actually computed this run |
| `ground_truth_cache_hit` / `ground_truth_lookup_s` | Cache status and lookup time for the keyed ground truth |
| `retrieval_batch_wall_time_s` | Elapsed wall time of the whole retrieval batch |
| `summed_query_execution_time_s` | Sum of the individually timed query executions (subset of the wall time) |
| `retrieval_total_s` | Legacy alias of the summed execution time |
| `retrieval_mean_per_query_ms` | Mean of the individual latencies |
| `metrics_computation_s` | Majority vote, scores and cross-repetition summary |
| `persistence_s` | Experiment, repetition and latency inserts |
| `orchestration_overhead_s` | Residual: loading, counting, fingerprint, loop overhead |

Only the non-overlapping phases (batch wall time, not the summed subset) are
subtracted from the total to form the residual. None of these fields is an HTTP
latency; that must be measured at the request boundary. The ground truth is
cached on pool fingerprint, query ids, k, metric and tolerance; 2 misses and 40
hits occurred in the final run.

Dataset-level duplicate and tie statistics, used to interpret Recall@K:

```powershell
.venv\Scripts\python -m scripts.dataset_statistics `
    --dataset-id 116 --k 5 --metric l2 --max-queries 3000 `
    --output ..\docs\results\duplicate_stats.json
```

---

## Tests

```powershell
cd backend
.venv\Scripts\python -m pytest
```

The suite contains **241 passing tests and 1 documented skip** (cosine
distance is discrete for collinear vectors, so a near-zero non-zero boundary
cannot be constructed). It covers classification/metric maths, tie-aware and
conventional Recall@K, tolerance validation for L2/cosine/L1, the leakage
contract of preprocessing, stratified splitting and query selection,
exhaustive balance checks for the supported benchmark matrix, exact
training-pool allocation for arbitrary integer seeds, timing/cache semantics,
and —
against a live database — exact/HNSW retrieval, deterministic tie-breaking,
index build/reuse bookkeeping, repetition accounting, persisted latency-row
counts, balanced execution order, timing decomposition and the batch runner's
CSV/JSON consistency.

---

## Further reading

| Document | Contents |
| --- | --- |
| [`docs/Thesis-Final-Verified.docx`](docs/Thesis-Final-Verified.docx) | **Authoritative current thesis** — aligned with the final v3 benchmark, figures and result artifacts |
| [`docs/archive/Thesis-KNN-PostgreSQL-Platform.docx`](docs/archive/Thesis-KNN-PostgreSQL-Platform.docx) | Superseded earlier thesis; preserved for history only, do not quote for v3 numbers |
| [`docs/Project-Guide.docx`](docs/Project-Guide.docx) | The function-by-function tour of the database, the backend and the frontend |
| [`docs/architecture.md`](docs/architecture.md) | The design decisions and the traps they avoid |
| [`docs/api.md`](docs/api.md) | The HTTP API reference |
| [`docs/experiment-plan.md`](docs/experiment-plan.md) | The experimental design and the threats to validity |
| [`docs/results/README.md`](docs/results/README.md) | The measured results and their interpretation |

The authoritative thesis and the project guide are generated from source by
the scripts in `docs/tools/` and can be regenerated whenever the project
changes:

```powershell
cd backend
.venv\Scripts\python ..\docs\tools\build_final_thesis.py
.venv\Scripts\python ..\docs\tools\build_guide.py
.venv\Scripts\python ..\docs\tools\verify_snippets.py ..\docs\Project-Guide.docx
```

`build_thesis.py` is a legacy builder for the archived, superseded thesis and
must not be used for current benchmark claims.

`verify_snippets.py` checks that every Python and TypeScript listing in the
documents still matches the source tree, and reports any divergence.
