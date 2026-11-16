# Experiment Plan

This document maps the research questions from the project proposal onto the
experiments the platform can run, and states the expected observations.

## Datasets

| Dataset | Rows | Features | Classes | Source |
| --- | --- | --- | --- | --- |
| Forest CoverType | 581,012 | 54 | 7 | UCI ML Repository (seeded into the platform) |
| *Any uploaded CSV* | user-defined | user-defined | user-defined | CSV upload |

Covertype is downloaded once from the UCI archive and cached under
`backend/data/`. It is cached in gzip form (~11 MB).

## Preprocessing contract

1. **Stratified** train/test split, preserving the class distribution.
2. The scaler (mean / standard deviation) and the categorical value mappings
   are **fitted on the training split only**. This is enforced by the code
   structure — `fit_preprocessor` never sees the test rows — and covered by
   the `test_leakage_contract` test.
3. All features are scaled together, so no feature dominates the distance
   computation.
4. The fitted state is stored as JSON on the dataset row, which makes every
   experiment reproducible and auditable.

## Experiments

### E1 — Exact baseline

Establishes the reference accuracy, macro F1 and per-query latency for a
given dataset, `k` and distance metric.

### E2 — `ef_search` sweep

For a fixed dataset, `k` and metric, run HNSW at a range of `ef_search`
values (e.g. 5, 10, 25, 50, 100, 200, 400, 800) and record:

* Recall@K against the exact ground truth
* Classification accuracy and macro F1
* p50 / p95 latency
* Whether the planner used the HNSW index

**Expected observation:** Recall@K increases monotonically with `ef_search`
and converges to 1.0, while latency increases roughly linearly. At very high
`ef_search` the planner may switch back to a sequential scan, at which point
the results converge exactly onto the exact baseline.

### E3 — Dataset-size scaling

Repeat E1 and E2 with the training pool capped at several sizes (e.g. 5k,
15k, 50k, 150k, 581k rows) using stratified subsampling.

**Expected observation:** exact-search latency grows roughly linearly with the
table size, whereas HNSW latency grows much more slowly (logarithmic in the
number of rows). The relative speed-up therefore widens as the dataset grows,
which is the practical argument for approximate retrieval.

### E4 — Distance metric sensitivity

Repeat the comparison for `l2`, `cosine` and `l1`.

**Expected observation:** cosine distance requires normalised vectors for a
meaningful interpretation; the platform scales features to zero mean and unit
variance, which approximates but does not guarantee unit L2 norm.

### E5 — Neighbourhood size

Sweep `k` (e.g. 1, 3, 5, 7, 11, 21).

**Expected observation:** larger `k` makes the majority vote more stable, so
the accuracy gap between exact and approximate retrieval narrows even when
Recall@K is unchanged.

## Metrics

| Metric | Definition |
| --- | --- |
| Accuracy | `correct / total` |
| Macro precision / recall / F1 | Unweighted mean over the 7 classes |
| Recall@K | Mean fraction of the exact top-k neighbours that were retrieved |
| Latency avg / p50 / p95 / p99 | Wall-clock time of a single retrieval query, measured server-side |
| Index build time | Duration of `CREATE INDEX ... USING hnsw` |

## Reproducing

```powershell
cd backend

# Seed a dataset
curl.exe -X POST http://127.0.0.1:8000/api/datasets/covertype `
  -H "Content-Type: application/json" `
  -d '{"name":"Forest CoverType","test_size":5000}'

# Run the matrix
.venv\Scripts\python -m scripts.run_experiments `
    --dataset-id 1 `
    --sizes 5000,15000,581000 `
    --ef-search 10,25,50,100,200,400 `
    --k 5 --max-queries 2000 `
    --output ..\docs\results\sweep.csv `
    --json ..\docs\results\sweep.json
```

## Threats to validity

* **Class imbalance.** CoverType is heavily imbalanced (class 2 has ~211k
  rows, class 4 has ~2.7k). Accuracy therefore over-represents the majority
  classes; macro F1 is reported alongside it for that reason.
* **Warm cache.** Measurements are taken after a warm-up query, so the
  PostgreSQL buffer cache is populated. Cold-cache behaviour is not measured.
* **Single machine.** All timings come from one machine; the absolute values
  are not a claim about production hardware, only about the *relative*
  difference between the two retrieval strategies.
* **Query-per-request overhead.** Each query is issued separately so that a
  per-query latency can be measured; a batched workload would amortise the
  round-trip and change the speed-up ratio.
