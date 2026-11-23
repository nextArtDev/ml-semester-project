# Phase 3 — Scientific Benchmark Protocol

**Branch:** `phase3-benchmark-design`  
**Based on:** `phase2-remediation` (`f10e7dc`)  
**Status:** design only — **benchmark not executed**  
**Approval required before execution**

---

## 1. Research objective

Investigate the trade-offs between **Exact KNN** and **Approximate HNSW KNN** for supervised classification using PostgreSQL 18.0 and pgvector 0.8.7, across retrieval accuracy, classification performance, query latency, index construction cost, storage overhead, training-pool size and HNSW search effort.

This protocol is the specification for one future run. It does not execute that run.

## 2. Research questions and hypotheses

| ID | Research question | Hypothesis (pre-registered) | Mode |
|---|---|---|---|
| RQ1 | How does `ef_search` affect Recall@5 and neighbour agreement? | Recall@5 and top-5 overlap with exact neighbours increase with effort, with diminishing returns. Non-monotonicity is a legitimate result, not an error. | Natural |
| RQ2 | How does retrieval approximation affect classification quality? | Accuracy and macro-F1 approach the exact baseline as `ef_search` increases; the gap is smaller at small `k` and larger pools. | Natural |
| RQ3 | What is the latency trade-off? | HNSW reduces p50/p95 latency relative to exact scan, and the relative advantage grows with training-pool size. | Natural |
| RQ4 | How do build time and storage scale? | HNSW build time and index size grow with pool size; with fixed `m`/`ef_construction`, stored index size is roughly proportional to rows. | Natural (metadata) |
| RQ5 | Can planner fallback invalidate a nominal HNSW result? | At the approved pool sizes, natural plans should select HNSW for all tested `ef`; any fallback must be excluded from HNSW summaries. | Natural + validity check |
| RQ6 | What changes when index selection is explicitly encouraged? | Controlled HNSW can obtain index scans where the natural planner would fall back, but its latency/quality must be reported separately from natural results. | Controlled diagnostic only |

Hypotheses are two-sided where direction is not guaranteed. The analysis must not be tuned to produce significance.

## 3. Audit of the existing implementation

Read-only audit of the current implementation on `phase2-remediation`:

| Area | Finding |
|---|---|
| Dataset preparation | Parent dataset 116 has 50,000 training rows, 3,000 held-out rows and 54 dimensions in the live database. `build_preprocessed_variant()` creates immutable derived datasets by stratified largest-remainder selection with `random.Random(f"{seed}:{label}")`; 10,000 and 50,000 rows are supported. The 50,000-row variant uses all available parent training rows and is a fresh refit/copy, not an independent sample. |
| Train/test separation | `fv_<id>` and `tv_<id>` are separate tables with typed `vector(54)` columns. Variant preprocessing is fitted only on selected training rows; test vectors are transformed with that fitted state. The runner persists fingerprints, class distributions, selected-id hashes and preprocessing-state hashes. |
| Feature preprocessing | `fit_preprocessor()` learns imputation/category/scaler state from the supplied training frame only. Covertype mode standardises continuous features and leaves 0/1 indicators unscaled. Test leakage is covered by existing tests and documented in `docs/experiment-plan.md`. |
| Exact KNN | Uses the pgvector **function** form `l2_distance(embedding, %s::vector)` with `ORDER BY distance, id LIMIT k`, which cannot use HNSW and is exhaustive by construction. Also supplies tie-tolerant ground truth. |
| HNSW KNN | Uses the **operator** form `embedding <-> %s` with `vector_l2_ops`; verifies the exact index name, access method, opclass and validity flags; applies `hnsw.ef_search` on the executing connection with `SET`/`SHOW`; restores the previous value; verifies the plan for every timed query; marks non-index runs `failed`. |
| Classification / voting | `majority_vote()` is unweighted by default (`weighted=False`), with ties broken by lower encoded label. Both arms use the same function. |
| Metrics | `accuracy`, macro `precision_macro`, `recall_macro`, `macro_f1`, per-class precision/recall/F1/support, confusion matrix, conventional `recall_at_k`, tie-aware `tie_aware_recall_at_k`, and latency percentiles are implemented. |
| Repetitions | `--repetitions N` executes N complete timed passes over the full query set; one warm-up query is discarded; `query_latencies` stores `N × queries` rows with `repetition` and `query_index`. |
| Timing boundaries | `time.perf_counter()` around individual `execute + fetch`; `summed_query_execution_time_s` is a subset of `retrieval_batch_wall_time_s`; index build, ground truth, metrics, persistence and orchestration are separated. EXPLAIN and plan checks run after the timer stops. |
| Artifact generation | Runner emits `sweep.csv/json`, `repetition_details.csv`, `repetition_summary.csv`, `query_samples.json`, `benchmark_orders.json` and `variant_ids.json`. `verify_results.py` validates artifact/DB agreement and rejects invalid HNSW rows. |

No defect that blocks protocol design was found. The Phase 2 remediation is a prerequisite and is present at the base commit.

## 4. Experimental variables

**Independent variables**

| Variable | Levels |
|---|---|
| Training-pool size | 10,000; 50,000 rows |
| Retrieval method | `exact`; `hnsw` |
| HNSW `ef_search` | 10; 50; 100; 400 (HNSW only) |
| Planner mode | `natural` (primary); `controlled_hnsw` (separate diagnostic) |
| Repetition | 1..3 within each run |
| Round | 1..3 (seeded balanced permutations) |

**Dependent variables:** retrieval metrics, classification metrics, latency statistics, throughput, index build time, index size and storage-overhead ratio.

**Controlled variables**

| Variable | Fixed value |
|---|---|
| Parent dataset | 116 (`Covertype 50k`), 54 dimensions, 7 classes |
| Held-out pool | 3,000 rows per variant |
| Evaluated queries | all 3,000 (`--max-queries 0`), stratified sample seed 0 |
| `k` | 5 |
| Distance metric | L2 (`<->` / `vector_l2_ops`) |
| HNSW build parameters | `m=16`, `ef_construction=64` (existing defaults; no change) |
| Index lifecycle | built once per size on first HNSW run, reused for all ef values and rounds |
| Voting | unweighted majority vote, deterministic lower-label tiebreak |
| Training sub-sample seed | 0 |
| Benchmark order seed | 0 |
| Bootstrap seed | 20261123 |
| Cache state | warm buffer cache; no cold-cache procedure |
| Timing boundary | client-side `execute + fetch` per query; plan checks excluded |

Within a pool size, exact and HNSW use the same derived dataset, query ids, query vectors, preprocessing state, metric, k and voting rule. Across pool sizes, preprocessing is deliberately refitted on that size's training rows; this is part of the size treatment and is documented, not a hidden confound.

## 5. Experiment matrix

### 5.1 Primary natural-planner benchmark

| Size | Configurations | Rounds | Repetitions | Runs | Timed queries |
|---:|---|---:|---:|---:|---:|
| 10,000 | exact, HNSW ef=10/50/100/400 | 3 | 3 | 15 | 135,000 |
| 50,000 | exact, HNSW ef=10/50/100/400 | 3 | 3 | 15 | 135,000 |
| **Total** | 5 configurations | 3 | 3 | **30** | **270,000** |

Run count = 2 sizes × 5 configurations × 3 rounds. Each configuration therefore has 9 complete passes over the same 3,000 query ids.

The runner's balanced ordering guarantees a seeded permutation per round; with 5 configurations and 3 rounds, positional balance is acceptable and is exported in `benchmark_orders.json`. Any HNSW run with a non-index access path is persisted as `failed` and excluded from every valid summary; it is reported as a cell-level finding for RQ5, not relabelled.

### 5.2 Separate controlled-HNSW diagnostic

| Size | Configurations | Repetitions | Runs |
|---:|---|---:|---:|
| 10,000 | HNSW ef=10/50/100/400 | 3 | 4 |
| 50,000 | HNSW ef=10/50/100/400 | 3 | 4 |
| **Total** | | | **8** |

These runs use the same derived datasets from `variant_ids.json`, `planner_mode='controlled_hnsw'`, `BEGIN`, `SET LOCAL enable_seqscan=off`, transaction-local ef and `ROLLBACK`. They carry `run_id='phase3-controlled-diagnostic-v1'` and are never pooled with natural-mode results or speed-ups.

### 5.3 Index parameters

`m=16` and `ef_construction=64` are the current defaults in `ExperimentConfig`, `_build_hnsw_index()` and the runner CLI. They are passed explicitly and recorded from the live index in `hnsw_m` / `hnsw_ef_construction`. No alternate index build configuration is part of Phase 3.

## 6. Fair-comparison rules

1. **Identical training data within a size:** one immutable derived dataset per size; exact and all ef values query `fv_<variant>`.
2. **Identical query ids within a size:** all 3,000 ids from `tv_<variant>`, selected by stratified seed 0 and exported in `query_samples.json`.
3. **Identical preprocessing within a size:** single fitted `PreprocessState` per variant, used for both arms.
4. **Identical metric and k:** L2, k=5 everywhere.
5. **Identical decision rule:** unweighted majority vote for both arms.
6. **Consistent timing:** same query timer for exact and HNSW; EXPLAIN/plan checks after timer; warm-up before first timed query only.
7. **No test leakage:** preprocessing state fitted on training rows only; labels used only for final evaluation.
8. **Order control:** seeded balanced rounds; exact is not always first.
9. **Mode separation:** `planner_mode` is persisted; controlled results are diagnostic only.
10. **Validity gate:** `status='completed'`, `used_index_scan=true` for HNSW, and `valid_for_summary=true` are required for a result to enter summaries.

## 7. Metrics

### Retrieval

- **Recall@5:** mean `|retrieved ∩ exact top-5 ids| / 5` per query.
- **Tie-aware Recall@5:** against the documented boundary-candidate set (`rel=1e-6`, `abs=1e-6`).
- **Neighbour agreement:** per-query top-5 Jaccard index and ordered overlap against exact ids, computed from `query_latencies.retrieved_ids`. This is a post-run derived metric; no new schema column is needed.

### Classification

- Accuracy.
- Macro precision, macro recall, macro F1.
- Per-class precision/recall/F1/support.
- Confusion matrix.

### Performance

- p50, p95, p99, min and max latency from all stored observations.
- End-to-end batch throughput = `unique_queries × repetitions / retrieval_batch_wall_time_s` (includes Python loop/row conversion; labelled as end-to-end, not pure DB throughput).
- Index construction time (`index_created_ms` on fresh builds; `index_reused_build_ms` retained separately).
- Index size (`pg_relation_size`), and storage-overhead ratio = index size / table heap size, computed post-run from `pg_relation_size('fv_<variant>')`.

## 8. Statistical methodology

- **Unit of inference:** query id within a pool size. Repetitions and rounds are repeated measurements of the same queries, not independent population samples.
- **Aggregation:** for each configuration × query, aggregate the 9 passes (3 rounds × 3 repetitions) using the median for latency and majority across passes for classification correctness; retain all raw samples for sensitivity analysis.
- **Latency:** primary test is two-sided Wilcoxon signed-rank on paired per-query HNSW−Exact latency differences. Report median difference with a 95% percentile bootstrap CI over query IDs (10,000 resamples, seed 20261123). Secondary sensitivity: paired t-test on log-latency.
- **Retrieval:** paired bootstrap CI for mean Recall@5 and Jaccard differences; McNemar's test is not used for recall because it is not binary per query.
- **Classification:** McNemar's exact test on paired per-query correctness (HNSW vs exact) and paired bootstrap CI for macro-F1 difference computed by resampling query IDs.
- **Multiple comparisons:** four ef levels per pool size form one family; Holm correction within that family. No correction across pool sizes or across metric families; all comparisons are reported.
- **Effect sizes:** median paired difference, rank-biserial correlation for Wilcoxon, and bootstrap CI bounds. Significance alone is never the criterion for a claim.
- **Outliers:** never deleted. Report min, max and p99. A 1%/99% winsorized latency analysis is a labelled sensitivity check only.
- **Run-to-run variability:** report t-based 95% CI across the 3 rounds separately and describe it as timing/order variability, not population error.
- **Cache effects:** warm cache for every configuration; one warm-up query per experiment; positional balance across rounds. Cold-cache and batch-throughput claims are out of scope.

## 9. Execution modes by research question

| Research question | Mode | Reason |
|---|---|---|
| RQ1 retrieval accuracy | Natural | Quality of the planner-selected production path. |
| RQ2 classification | Natural | Same reason. |
| RQ3 latency trade-off | Natural | Controlled forcing would create an artificial execution path. |
| RQ4 build time / storage | Natural metadata | Build/storage do not depend on planner mode. |
| RQ5 fallback detection | Natural + validity gate | A fallback is a result about natural planner behavior. |
| RQ6 forced-index sensitivity | Controlled diagnostic only | Must not be combined with natural results. |

## 10. Expected runtime and resources

Based on the archived 42-run benchmark (`docs/results/sweep.json`, total 754.6 s, mean 18.0 s/run, max 102.6 s/run), the 30-run natural matrix is expected to take roughly **10–20 minutes**, plus derived-variant creation and first-time index builds. The 8 controlled diagnostic runs add roughly **3–8 minutes**. Plan for **20–40 minutes** end to end.

| Resource | Requirement |
|---|---|
| PostgreSQL | localhost:5432, database `knn_knn_platform`, pgvector 0.8.7, superuser or equivalent for variant DDL |
| Python | `backend/.venv` with numpy, pandas, psycopg, psycopg-pool, pydantic-settings, FastAPI/pytest |
| Memory | `maintenance_work_mem='512MB'` during HNSW build (session-local in code); `work_mem=4MB` for queries |
| Disk growth | Two derived variants (~3 MB + ~14 MB heap, ~5 MB + ~26 MB index) plus ~270k latency rows; order 50–150 MB; verify free space before execution |
| Backup | Mandatory verified `pg_dump -Fc` before execution; no destructive migration |
| Historical data | Do not delete or backfill experiments 382–384 or any prior artifact |

## 11. Exact execution commands

All commands use Git Bash from the repository root. Confirm the branch and commit first:

```bash
cd "C:/Users/aria/Desktop/NextJS Projects/ml-semester-project"
git status --short --branch
git rev-parse HEAD
```

### 11.1 Backup

```bash
cd backend
export PGPASSWORD="$(.venv/Scripts/python.exe -c "from dotenv import dotenv_values; print(dotenv_values('.env')['POSTGRES_PASSWORD'])")"
mkdir -p "C:/Users/aria/Documents/ml-semester-project-db-backups"
"/c/Program Files/PostgreSQL/18/bin/pg_dump.exe" -Fc -h localhost -p 5432 -U postgres -d knn_knn_platform -f "C:/Users/aria/Documents/ml-semester-project-db-backups/knn_knn_platform_phase3_pre.dump"
"/c/Program Files/PostgreSQL/18/bin/pg_restore.exe" --list "C:/Users/aria/Documents/ml-semester-project-db-backups/knn_knn_platform_phase3_pre.dump" > "C:/Users/aria/Documents/ml-semester-project-db-backups/knn_knn_platform_phase3_pre.list.txt"
sha256sum "C:/Users/aria/Documents/ml-semester-project-db-backups/knn_knn_platform_phase3_pre.dump"
```

### 11.2 Primary natural-planner benchmark

```bash
cd backend
mkdir -p ../docs/results/phase3
.venv/Scripts/python.exe -m scripts.run_experiments \
  --dataset-id 116 \
  --sizes 10000,50000 \
  --planner-mode natural \
  --ef-search 10,50,100,400 \
  --k 5 --metric l2 \
  --max-queries 0 \
  --repetitions 3 --rounds 3 \
  --order-seed 0 --seed 0 \
  --query-sampling stratified --query-sample-seed 0 \
  --hnsw-m 16 --hnsw-ef-construction 64 \
  --run-id phase3-natural-v1 \
  --output ../docs/results/phase3/sweep.csv \
  --json ../docs/results/phase3/sweep.json \
  --repetition-details ../docs/results/phase3/repetition_details.csv \
  --repetition-summary ../docs/results/phase3/repetition_summary.csv \
  --query-samples ../docs/results/phase3/query_samples.json \
  --benchmark-orders ../docs/results/phase3/benchmark_orders.json \
  --variant-ids ../docs/results/phase3/variant_ids.json
```

### 11.3 Verify primary artifacts

```bash
cd backend
.venv/Scripts/python.exe -m scripts.verify_results \
  --results-dir ../docs/results/phase3 \
  --run-id phase3-natural-v1 \
  --expected-runs 30 --expected-rounds 3 \
  --expected-queries 3000 --expected-repetitions 3
```

### 11.4 Separate controlled-HNSW diagnostic

```bash
cd backend
.venv/Scripts/python.exe - <<'PY'
import json
from pathlib import Path

from app.services.experiments import ExperimentConfig, run_experiment
from app.services.metrics import DistanceMetric

art = Path("../docs/results/phase3")
variants = json.loads((art / "variant_ids.json").read_text(encoding="utf-8"))["variants"]
rows = []
for size in (10000, 50000):
    variant = next(v for v in variants.values() if int(v["target_size"]) == size)
    dataset_id = int(variant["dataset_id"])
    for ef in (10, 50, 100, 400):
        result = run_experiment(
            ExperimentConfig(
                dataset_id=dataset_id,
                k=5,
                distance_metric=DistanceMetric.L2,
                method="hnsw",
                ef_search=ef,
                planner_mode="controlled_hnsw",
                hnsw_m=16,
                hnsw_ef_construction=64,
                max_queries=3000,
                repetitions=3,
                seed=0,
                query_sampling="stratified",
                query_sample_seed=0,
                run_id="phase3-controlled-diagnostic-v1",
                name=f"controlled-hnsw size={size} ef={ef}",
            )
        )
        rows.append({
            "run_id": "phase3-controlled-diagnostic-v1",
            "target_size": size,
            "dataset_id": dataset_id,
            "ef_search": ef,
            "experiment_id": int(result["id"]),
            "status": result["status"],
            "used_index_scan": result["used_index_scan"],
            "error_message": result.get("error_message"),
            "latency_p50_ms": result["latency_p50_ms"],
        })
(art / "controlled_diagnostic.json").write_text(
    json.dumps(rows, indent=2), encoding="utf-8"
)
print(f"wrote {art / 'controlled_diagnostic.json'}")
PY
```

The controlled diagnostic creates additional experiment rows on the same immutable variants. They are identifiable by `planner_mode='controlled_hnsw'` and the diagnostic `run_id`; they are not included in `sweep.csv` or any natural-mode summary.

### 11.5 Post-run analysis (after approval of execution)

The paired analysis and neighbour-agreement metrics require a derived-analysis script. It must read only the Phase 3 artifacts and database views, apply the statistical rules in section 8, and write a separate report. It is intentionally not specified as executable here because execution is not approved.

## 12. Acceptance criteria for this protocol

- Internally consistent: two sizes, five primary configurations, three rounds, three repetitions, one k/metric/voting rule; natural and controlled modes separated.
- Reproducible: parent dataset 116, seed 0, order seed 0, explicit index parameters, exported query ids/orders/variant identities, exact commands.
- Statistically defensible: query-level paired comparisons, repeated measures treated correctly, pre-registered tests, Holm correction, CIs, outlier policy.
- Safe: verified backup, no historical backfill, no dataset 116 modification, no full benchmark before approval.

**Stop here. Do not execute the benchmark or thesis builder until this protocol is approved.**
