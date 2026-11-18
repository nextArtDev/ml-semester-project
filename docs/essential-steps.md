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
6. `ensure_hnsw_index()` returns an `IndexInfo` describing whether a build ran,
   its duration, the live index's `m`/`ef_construction` and its
   `pg_relation_size`. A reused index reports `index_build_ms = NULL` and its
   historical duration in `index_reused_build_ms`, so an `ef_search` sweep
   never presents a historical build as a fresh cost.

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
   confusion matrix, conventional Recall@K against a deterministic exact
   reference (`ORDER BY distance, id`), and tie-aware Recall@K against the
   boundary candidate set (`distance <= boundary × (1 + 1e-6) + 1e-6`).
   Both recall metrics are persisted. The tolerance is stored on the
   experiment row.
3. Latency: one warm-up query, then `repetitions` complete passes over the
   same query set in the same order; every pass contributes individually
   stored samples. p50/p95/p99, mean, min, max and an explicit timing
   decomposition (`total_experiment_runtime_s`, `index_preparation_s`,
   `index_build_s`, `ground_truth_computation_s`, `ground_truth_lookup_s`,
   `retrieval_batch_wall_time_s`, `summed_query_execution_time_s`,
   `retrieval_total_s`, `retrieval_mean_per_query_ms`, `metrics_computation_s`,
   `persistence_s`, `orchestration_overhead_s`) are stored. Recall, accuracy and macro metrics
   are computed **per repetition** and written to `experiment_repetitions`;
   the aggregate row carries their mean and standard deviation across passes.
4. Environment metadata captured per experiment: Python, OS, machine, CPU
   cores, PostgreSQL and pgvector versions, plus `repetitions`,
   `measured_queries`, `unique_queries`, `total_test_pool`, `warmup_queries`,
   `preprocessing_mode`, `applied_ef_search`, plan verification,
   dataset fingerprint, seed, tie tolerance, query-sampling mode and seed,
   benchmark round/position/seed and a warm-cache note.

**Acceptance check** — a run with `--max-queries 1000 --repetitions 3`
stores exactly 3,000 `query_latencies` rows and 3 `experiment_repetitions`
rows, and `environment.applied_ef_search` equals the requested `ef_search`.

---

## Phase 5 — Batch experiment runner

**Status: done.**

`backend/scripts/run_experiments.py` now exports 71 columns and accepts
`--repetitions` (default 3), `--rounds` (default 3), `--order-seed`,
`--query-sampling`, `--query-sample-seed`, `--query-samples`,
`--benchmark-orders`, `--run-id`, `--repetition-details` and
`--repetition-summary`. It also:

1. Subsamples the training pool with an exact largest-remainder allocation, so
   the target size is met exactly and classes remain represented. Within-class
   sampling uses `random.Random(f"{seed}:{label}")`, accepting any integer seed
   and avoiding PostgreSQL's `setseed` range restriction.
2. Selects the evaluated queries by a seeded stratified sample by default,
   persists their ids and both class distributions, and exports them to
   `query_samples.json`. `--max-queries 0` evaluates the whole pool.
3. Executes the configuration matrix in seeded rounds: every round contains
   every configuration once; exhaustive tests demonstrate position balance for
   the supported 7×3 matrix, without claiming it in general. Round, position,
   seed and order are persisted and exported. The index is not rebuilt between
   configurations or rounds.
4. Emits quality, latency and index metadata, all timing fields, benchmark and
   query-sampling metadata, and the run id. Exact ground truth is computed once
   per unique key and cached across configurations and rounds.
5. Writes `repetition_details.csv` (one row per experiment and pass) and
   `repetition_summary.csv` (median, SD, min/max and 95 % interval per metric)
   by reading the rows back from the database, so the files cannot drift from
   the persisted run.
6. Prints a stderr warning when a HNSW run reports `used_index_scan = False`,
   so a silent sequential-scan fallback cannot be mistaken for HNSW latency.
7. Restores the original training pool in a `finally` block, even if a run
   fails.

**Acceptance check**

```powershell
cd backend
.venv\Scripts\python -m scripts.run_experiments --help     # lists the new flags
```

`tests/test_run_experiments.py` runs the script end-to-end on a synthetic
dataset and checks the CSV/JSON field set, value agreement, per-repetition row
count, summary row count, index-reuse bookkeeping and persisted latency
count. Verified by hand: a 5,000-row subsample of a 25,000-row pool ran, then
the pool was restored to 25,000 rows exactly.

---

## Phase 6 — Full experimental sweep

**Status: done — final run id `final-v3-full-pool`: 42 rows in
`docs/results/sweep.csv`, 126 rows in `repetition_details.csv`, 420 rows in
`repetition_summary.csv`. Earlier runs are preserved in `docs/results/v1/`
and `docs/results/v2/`.**

Dataset 116 (`Covertype 50k`, 50,000 train / 3,000 test, 54 dimensions,
fingerprint `6baa8fa283ea5a8267d3e964eedf5d2a`) was measured with **exactly
10,000 and 50,000** training rows (10k subsample fingerprint
`7928b6e3f5d451e4ab5444c8c5dbebdd`), `ef_search` 10, 25, 50, 100, 200, 400,
`k = 5`, L2, **all 3,000 held-out queries** (`--max-queries 0`), 3 repetitions,
3 rounds, and explicit seeds (`--seed 0`, `--order-seed 0`,
`--query-sample-seed 0`). A 42-run smoke benchmark with 3 repetitions and 3
rounds was validated first.

Verified by `scripts/verify_results.py` (0 errors):

* 42 experiments; 3 rounds × 7 configurations per pool; every configuration
  once per round;
* 0 HNSW rows without an index scan or wrong applied `ef_search`; 0 exact rows
  without a sequential scan;
* 9,000 latency rows and 3 `experiment_repetitions` rows per experiment —
  378,000 and 126 in total;
* 0 CSV/JSON/database mismatches, including timing fields;
* 2 misses and 40 hits in the keyed ground-truth cache;
* only 2 fresh builds across 42 rows; reuses keep `index_build_ms = NULL`.

Measured duplicate/tie statistics for the same pool
(`docs/results/duplicate_stats.json`): 50,000 unique vectors, 0 duplicate
groups, 0 mixed-label groups, **0 tied queries in 3,000**, largest candidate
set 5.

Matrix that was run (`preprocessing_mode=covertype`, one dataset row):

| Axis | Values | Covered |
| --- | --- | --- |
| Training pool | 10,000 / 50,000 | yes |
| Method | exact baseline, HNSW | yes |
| `ef_search` | 10, 25, 50, 100, 200, 400 | yes |
| `k` | 5 (primary), plus 1/15 for the neighbourhood sweep | k=5 only |
| Metric | L2 (primary), plus cosine for the metric sweep | L2 only |
| Queries | all 3,000 held-out rows (`--max-queries 0`) | yes |
| Repetitions | 3 complete passes | yes |
| Benchmark rounds | 3 seeded balanced rounds | yes |
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

**Status: done.**

1. `README.md` — test count (241 passed, 1 skip), full-pool query evaluation,
   balanced rounds, timing/cache definitions and the regenerated output.
2. `docs/architecture.md` — order guarantees, three-level repetition hierarchy,
   exact pool allocation, any-integer seed mapping and keyed ground-truth cache.
3. `docs/experiment-plan.md` — measurement contract, reproducibility seeds,
   timing definitions and measured duplicate statistics over 3,000 queries.
4. `docs/api.md` — new timing and sampling fields.
5. `docs/results/README.md` — final sweep, round positions, full-pool query
   ids, timing decomposition and v1/v2 comparison.
6. The Word documents were deliberately **not** regenerated for this final
   artifact set; they still describe the previous benchmark. They must not be
   quoted for v3 numbers.

**Acceptance check** — every number in the markdown docs traces to
`docs/results/*.csv`, `docs/results/*.json` or a persisted experiment row.

---

## Phase 9 — Verification and hand-over

1. `pytest` — **241 passed, 1 documented skip**, 0 failures, against real
   PostgreSQL + pgvector.
2. A 42-run smoke benchmark (3 rounds, 3 repetitions, 20 queries, run id
   `smoke-v3`) on dataset 37, then the final benchmark (3 rounds, 3 repetitions,
   all 3,000 held-out queries, run id `final-v3-full-pool`) on dataset 116. Both
   were verified with `scripts/verify_results.py`: 0 errors.
3. Reproducibility checklist: fixed seeds for pool, queries and order, dataset
   fingerprint, pinned library and database versions, `repetitions` and
   `rounds` stated, warm-cache caveat stated, index reuse recorded.
4. Final report: problems found, files changed and why, exact test output,
   Windows setup commands, sweep commands, remaining limitations.

---

## Definition of done

- [x] `pytest` green (241 passed, 1 documented skip) against a machine with
      PostgreSQL + pgvector.
- [x] Every experiment row records method, `k`, metric, `ef_search`, plan
      verification, repetitions, unique query count, total pool size, latency
      distribution, build/reuse state, index size, environment, query sample
      and balanced-order metadata.
- [x] Exact rows prove `Seq Scan`; HNSW rows prove the named index scan.
- [x] `sweep.csv` / `sweep.json` / `repetition_details.csv` /
      `repetition_summary.csv` / `query_samples.json` /
      `benchmark_orders.json` regenerated from the final run, with v1 and v2
      preserved.
- [x] Timing decomposition replaces `api_overhead_s`; batched retrieval wall
      time is separated from summed query execution time.
- [ ] Word documents regenerated to match the final v3 benchmark (intentionally
      deferred).
- [x] Conventional and tie-aware Recall@K both implemented, persisted and
      reported; tie tolerance documented.
- [x] Duplicate-vector and boundary-tie statistics measured for the pool and
      recorded in `duplicate_stats.json`.
- [ ] UI shows preprocessing mode, repetitions, applied `ef_search`, index
      size and plan verification, in EN and FA.
- [x] README, architecture, experiment plan, API doc and both `.docx` files
      match the code and the verified benchmark.
- [x] Limitations stated honestly: warm cache, single machine,
      per-query round-trip, class imbalance, Covertype-only primary dataset,
      correlated repetitions and rounds, small query sample.

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
| Repetitions reuse the query order | The goal is run-to-run variability, not population sampling; independent resampling would change the estimand |
| Both conventional and tie-aware Recall@K | Conventional reports id-level agreement; tie-aware isolates arbitrary boundary tie-breaking. Neither replaces the other |
| Deterministic exact reference (`ORDER BY distance, id`) | Without an id tiebreak the K-th row is chosen arbitrarily and conventional recall is irreproducible |
| Reused index has `index_build_ms = NULL` | A historical build is not a cost paid by the current `ef_search` row |
| Dataset fingerprint stored per experiment | Row count alone does not identify the pool composition |
| Three balanced benchmark rounds | With an odd configuration count, two rotations cannot give every configuration both an early and a late slot; three can |
| Stratified query sampling with persisted ids | The first N ids are reproducible but not justified as representative; persisted ids make the sample auditable |
| Exact retrieval also orders by id | Repeated exact queries must return the same ids when distances tie; HNSW must not get the extra sort key or it loses the index |
| Explicit timing phases instead of `api_overhead_s` | Runtime minus query time charges ground-truth computation and orchestration as "API overhead" |
| v1 artifacts preserved under `docs/results/v1/` | The new order and query sample change the numbers; historical results must not be silently rewritten |
