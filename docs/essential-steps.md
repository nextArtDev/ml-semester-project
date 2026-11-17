# Essential Steps

A build-order roadmap that maps the project proposal onto this repository:
what must exist, what already exists, and the exact order in which the
remaining work has to happen.

Each step has an **acceptance check** — a command or observation that proves
the step is finished. A step is not done because the code compiles; it is done
when the check passes on the real database.

---

## 0. Deliverables the proposal commits to

| # | Proposal commitment | Where it lives |
| --- | --- | --- |
| D1 | Working web platform (Next.js + FastAPI + PostgreSQL) | `frontend/`, `backend/` |
| D2 | Feature vectors and labels stored in PostgreSQL via pgvector | `fv_<id>` / `tv_<id>` tables |
| D3 | Exact KNN baseline | `retrieve(..., method="exact")` |
| D4 | HNSW approximate KNN | `retrieve(..., method="hnsw")` |
| D5 | Classification metrics + Recall@K + latency + build time | `services/metrics.py`, `services/experiments.py` |
| D6 | `ef_search` sweep and dataset-size sweep | `scripts/run_experiments.py` |
| D7 | Execution-plan proof that the intended method was used | `EXPLAIN (ANALYZE, BUFFERS)` captured per run |
| D8 | Final report with results, limitations, trade-offs | `docs/*.docx` |

The evaluation section (§5.6) additionally requires: accuracy, precision,
recall, macro F1, Recall@K, average / median / p95 latency, and HNSW build
time. The audit adds p99, index storage size, speed-up, repetition count and
environment metadata, because a single-pass timing cannot support a speed-up
claim.

---

## Phase 1 — Environment and database readiness

**Status: done.**

1. PostgreSQL 18 running on `localhost:5432`, database `knn_knn_platform`.
2. pgvector 0.8.7 loadable (extension control path + `dynamic_library_path`,
   plus a read/execute ACL grant for the `NT AUTHORITY\NETWORK SERVICE`
   account that PostgreSQL runs under).
3. Python virtual environment at `backend/.venv`.
4. Schema created idempotently on startup (`MIGRATION_SQL` uses
   `ADD COLUMN IF NOT EXISTS`, so an existing database upgrades in place).

**Acceptance check**

```powershell
curl.exe http://127.0.0.1:8000/api/health
# → {"status":"ok","database":"connected","pgvector_version":"0.8.7", ...}
```

---

## Phase 2 — Data layer

**Status: done.**

1. Covertype downloaded once from UCI and cached at
   `backend/data/covertype.csv.gz` (~11 MB, git-ignored). 55 columns:
   10 continuous + 4 wilderness + 40 soil + label.
2. `stratified_split()`:
   * `train_fraction` is a real fraction of rows (the earlier version treated
     it as "fraction of the *test* size", which silently produced 50/50
     splits);
   * validated to `0 < fraction < 1`, with a configurable seed;
   * largest-remainder proportional allocation, so the requested size is hit
     exactly instead of approximately;
   * rare classes (< `min_class_count`) default to the training split so no
     class loses all of its training examples.
3. `fit_preprocessor()` is fitted on the **training split only** and returns a
   serialisable `PreprocessState` (medians, modes, scaler mean/scale,
   one-hot vocabularies, `output_columns`, `label_to_index`).
4. Preprocessing contract:
   * continuous → median imputation → standardised;
   * binary indicators → mode imputation → kept as 0/1 in `covertype` mode
     (standardised only in the optional `standardize_all` mode);
   * nominal categorical → mode imputation → **one-hot**, sorted vocabulary,
     unseen categories become an all-zero block;
   * target is never a feature; train and test share dimension and column
     order by construction.
5. `validate_dataset()` rejects: empty frame, missing target column, missing
   target values, entirely-empty columns, `inf` values, a single class, and
   classes below `min_class_count`. The API maps these to HTTP 400.

**Acceptance check**

```powershell
cd backend
.venv\Scripts\python -m pytest tests\test_preprocessing.py tests\test_datasets.py -q
```

Plus: the ingested dataset row must report
`preprocessing_mode=covertype`, `dimension=54` (10 numeric + 44 binary).

---

## Phase 3 — Storage and retrieval

**Status: done.**

1. One pair of typed tables per dataset (`fv_<id>`, `tv_<id>`) with
   `embedding vector(<dim>)`. A fixed dimension is mandatory for HNSW, and a
   shared table would apply the `dataset_id` filter *after* the index scan,
   silently returning neighbours from other datasets.
2. **Exact retrieval uses the distance *function***
   (`l2_distance`, `cosine_distance`, `l1_distance`). Function form is not
   index-eligible, so the planner cannot silently turn the baseline into an
   approximate search.
3. **HNSW retrieval uses the distance *operator*** (`<->`, `<=>`, `<+>`),
   which is what the index satisfies.
4. `ef_search` is applied with a session-level `SET`, read back with
   `SHOW hnsw.ef_search`, and `RESET` in a `finally` block. `SET LOCAL` is a
   no-op on this pool because it runs `autocommit=True` — the first version of
   the sweep was silently pinned at the default 40.
5. Every retrieval captures `EXPLAIN (ANALYZE, BUFFERS)` and records
   `used_index_scan`, `used_sequential_scan` and `applied_ef_search`. The app
   therefore reports when HNSW was **not** used instead of assuming it was.
6. Index build time and on-disk index size are recorded
   (`pg_relation_size`), and the index is reused when metric and build
   parameters are unchanged.

**Acceptance check**

```powershell
cd backend
.venv\Scripts\python -m pytest tests\test_vector_store.py -q
```

And by hand: exact plan contains `Seq Scan`, HNSW plan contains
`Index Scan using fv_<id>_embedding_hnsw_idx`, and `applied_ef_search` equals
the requested value.

---

## Phase 4 — Classification and measurement

**Status: done.**

1. Majority voting over the returned neighbours, identical code path for both
   methods, so any accuracy difference is attributable to retrieval alone.
2. Metrics: accuracy, macro precision / recall / F1, per-class metrics,
   confusion matrix, Recall@K against exact ground truth.
3. Latency: warm-up query discarded, then `repetitions` full passes over the
   same query set; every pass contributes a latency sample. p50/p95/p99,
   mean, min, max, and `api_overhead_s` (end-to-end time minus query time) are
   stored separately.
4. Environment metadata captured per experiment: Python, OS, CPU, memory,
   NumPy / pandas / scikit-learn / psycopg / FastAPI versions, PostgreSQL and
   pgvector versions, plus `repetitions`, `measured_queries`,
   `warmup_queries`, `preprocessing_mode`, `applied_ef_search`.

**Acceptance check** — a run reports `repetitions=3` with
`measured_queries = queries × 3`, and `environment.applied_ef_search` equals
the requested `ef_search`.

---

## Phase 5 — Batch experiment runner

**Status: done.**

`backend/scripts/run_experiments.py` now exports 38 columns and accepts
`--repetitions` (default 3). It also:

1. Subsamples the training pool **stratified by label**, with a seeded
   `setseed`, so `--sizes` preserves class proportions instead of truncating
   by insertion order.
2. Emits `repetitions`, `precision_macro`, `recall_macro`, `latency_p99_ms`,
   `latency_min_ms`, `latency_max_ms`, `index_size_bytes`, `api_overhead_s`,
   `applied_ef_search`, `used_sequential_scan`, `speedup_p50`, `speedup_avg`,
   `preprocessing_mode` and the environment block.
3. Prints a stderr warning when a HNSW run reports `used_index_scan = False`,
   so a silent sequential-scan fallback cannot be mistaken for HNSW latency.
4. Restores the original training pool in a `finally` block, even if a run
   fails.

**Acceptance check**

```powershell
cd backend
.venv\Scripts\python -m scripts.run_experiments --help     # lists --repetitions
```

Verified: a 5,000-row subsample of a 25,000-row pool ran, then the pool was
restored to 25,000 rows exactly.

---

## Phase 6 — Full experimental sweep

**Status: done — 14 rows in `docs/results/sweep.csv`.**

Dataset 116 (`Covertype 50k`, `preprocessing_mode=covertype`, 50,000 train /
3,000 test, 54 dimensions) was ingested for this sweep. Training pools of
9,999 and 50,000 rows were measured at `ef_search` 10, 25, 50, 100, 200, 400,
`k = 5`, L2, 1,000 queries, 3 repetitions.

Verified after the run:

* 0 HNSW rows without an index scan;
* 0 exact rows without a sequential scan;
* 0 mismatches between requested and applied `ef_search`;
* 3,000 measured latencies per run (1,000 × 3);
* `preprocessing_mode = covertype` on every row.

Matrix that was run (`preprocessing_mode=covertype`, one dataset row):

| Axis | Values | Covered |
| --- | --- | --- |
| Training pool | 10,000 / 50,000 | yes |
| Method | exact baseline, HNSW | yes |
| `ef_search` | 10, 25, 50, 100, 200, 400 | yes |
| `k` | 5 (primary), plus 1/15 for the neighbourhood sweep | k=5 only |
| Metric | L2 (primary), plus cosine for the metric sweep | L2 only |
| Queries | 1,000 per run | yes |
| Repetitions | 3 | yes |
| HNSW build | `m=16`, `ef_construction=64` | yes |

The `k` sweep, the cosine metric sweep, a larger training pool and a
`standardize_all` comparison run are still open — they are optional axes of
the proposal, not prerequisites for the primary claim.

Rules that make the numbers defensible:

* exact and HNSW must use the **same** training pool, query set, transform,
  metric and `k`;
* Recall@K is always computed against the exact ground truth from the same
  pool;
* each result must record whether the index was actually used;
* do not assume HNSW is faster — report the measured speed-up, including the
  cases where it is not;
* every number in the report must be traceable to a persisted experiment row.

**Acceptance check** — `docs/results/sweep.csv` is regenerated, every HNSW row
has `used_index_scan = True`, and every exact row has
`used_sequential_scan = True`.

---

## Phase 7 — Frontend surfacing

**Status: partially done.**

Already added: `repetitions` input on the experiment runner, preprocessing-mode
selector on the dataset page, bilingual EN/FA strings, and the extended API
types.

Remaining:

1. Show index size, index build time, applied `ef_search`, repetition count,
   p99 / min / max latency and API overhead in the result panel.
2. Show the preprocessing mode for each dataset, and label the experiment as
   *verified index scan* or *fell back to sequential scan*.
3. Keep the change minimal — no redesign, no new pages.

**Acceptance check** — a completed run in the browser displays the applied
`ef_search` and a plan-verification badge, in both `/en` and `/fa`.

---

## Phase 8 — Documentation and deliverables

**Status: stale.**

1. `README.md` — claims "47 tests" and quotes the pre-audit numbers.
2. `docs/architecture.md` — still says "StandardScaler + ordinal encoding".
3. `docs/experiment-plan.md` — still says "all features are scaled together";
   must state the covertype vs `standardize_all` rationale and the
   no-silent-switching rule.
4. `docs/api.md` — must document `repetitions` and `preprocessing_mode`.
5. `docs/results/README.md` — replace the findings with the regenerated sweep.
6. Regenerate `Thesis-KNN-PostgreSQL-Platform.docx` and `Project-Guide.docx`
   through `docs/tools/build_thesis.py` / `build_guide.py`, then run
   `verify_snippets.py`, which checks that every code block in the document
   still matches the source file verbatim.

**Acceptance check** — `verify_snippets.py` reports zero mismatches and no
document contains a number that cannot be found in `docs/results/sweep.csv`.

---

## Phase 9 — Verification and hand-over

1. `pytest` — currently **123 passed**, 0 skipped, real PostgreSQL. Re-run
   after the runner and doc changes.
2. Reproducibility checklist: fixed seed, fixed splits, fixed query set,
   pinned library and database versions, `repetitions` stated, warm-cache
   caveat stated.
3. Final report: problems found, files changed and why, exact test output,
   Windows setup commands, sweep commands, remaining limitations.

---

## Definition of done

- [ ] `pytest` green with no skips on a machine with PostgreSQL + pgvector.
- [ ] Every experiment row records method, `k`, metric, `ef_search`, plan
      verification, repetitions, latency distribution, build time, index size
      and environment.
- [ ] Exact rows prove `Seq Scan`; HNSW rows prove the named index scan.
- [ ] `sweep.csv` / `sweep.json` regenerated from the corrected pipeline.
- [ ] UI shows preprocessing mode, repetitions, applied `ef_search`, index
      size and plan verification, in EN and FA.
- [ ] README, architecture, experiment plan, API doc and both `.docx` files
      match the code.
- [ ] Limitations stated honestly: warm cache, single machine,
      per-query round-trip, class imbalance, Covertype-only primary dataset.

---

## Decision log (do not silently revisit)

| Decision | Reason |
| --- | --- |
| Exact = distance function, not operator | The operator is index-eligible; the baseline would stop being exhaustive |
| Per-dataset vector tables | HNSW needs a declared dimension; residual filters run after the index scan |
| Session `SET` + `RESET` for `ef_search` | `SET LOCAL` is silently discarded when `autocommit=True` |
| `covertype` preprocessing is primary | Covertype's 44 indicators are already 0/1; standardising them destroys sparsity without benefit. `standardize_all` is an explicit, separately labelled comparison |
| HNSW only | IVFFlat was removed as out of scope for the core project |
| Repetitions over the same query set | A single pass cannot support percentile or speed-up claims |
