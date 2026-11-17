# Experiment Plan

This document maps the research questions from the project proposal onto the
experiments the platform can run, states the measurement contract, and lists
the expected observations as hypotheses rather than guaranteed outcomes.

## Datasets

| Dataset | Rows | Features | Classes | Source |
| --- | --- | --- | --- | --- |
| Forest CoverType | 581,012 | 54 | 7 | UCI ML Repository (seeded into the platform) |
| *Any uploaded CSV* | user-defined | user-defined | user-defined | CSV upload |

Covertype is downloaded once from the UCI archive and cached under
`backend/data/` in gzip form (~11 MB). The reported sweep uses a 50,000-row
training pool and a 3,000-row held-out pool (dataset id 116).

## Preprocessing contract

1. **Stratified** train/test split, preserving the class distribution.
2. The scaler (mean / standard deviation) and the categorical value mappings
   are **fitted on the training split only**. This is enforced by the code
   structure — `fit_preprocessor` never sees the test rows — and covered by
   the `test_leakage_contract` test.
3. In the primary `covertype` mode, the 10 continuous features are
   standardised to zero mean and unit variance, while the 44 wilderness/soil
   indicators are kept as 0/1. The indicators are already dimensionless and
   sparse; standardising them would spread each one across two non-zero
   values without changing which rows are close.
4. Arbitrary uploaded CSVs use median imputation, one-hot encoding of nominal
   columns, and standardisation of the resulting numeric block. Unseen
   categories map to an all-zero block instead of raising.
5. `standardize_all` is an explicit alternative that standardises every
   column, including the indicators. It is a separately labelled comparison,
   not the primary configuration, because it changes the distance geometry.
6. The fitted state is stored as JSON on the dataset row. The dataset
   fingerprint stored on each experiment covers the row count, the ordered
   `(id, label)` sequence and a checksum of the stored vector text, so a
   result can be tied to an exact pool composition.

## Measurement contract

The experiment distinguishes four quantities that are easy to conflate:

| Quantity | Definition | Example |
| --- | --- | --- |
| Held-out test pool | Size of the query pool | 3,000 |
| Unique evaluated queries | Queries actually executed, chosen once | 1,000 |
| Repetitions | Complete timed passes over that query set | 3 |
| Latency observations | Stored per-query samples | 3,000 |

`--repetitions N` executes N full passes; it never multiplies a counter. Every
sample is stored with `repetition` and `query_index`, and the stored count is
verified against `repetitions × unique_queries`. One warm-up query runs before
the first timed query and is not stored.

The query order is **identical** across repetitions. Repetitions therefore
measure run-to-run variability in timing and index traversal, not sampling
error over a population. Per-repetition metrics are persisted in
`experiment_repetitions`; the cross-repetition median, standard deviation and
95 % interval (Student's t with `n − 1` degrees of freedom) are exported in
`repetition_summary.csv`. Three repetitions are too few to support a narrow
interval, and the artifacts say so.

## Experiments

### E1 — Exact baseline

Establishes the reference recall, accuracy, macro F1 and per-query latency for
a given dataset, `k` and distance metric, using the pgvector distance
*function* so the plan is an exhaustive sequential scan by construction.

### E2 — `ef_search` sweep

For a fixed dataset, `k` and metric, run HNSW at a range of `ef_search`
values (e.g. 10, 25, 50, 100, 200, 400) and record:

* Conventional Recall@K against the deterministic exact neighbour ids
* Tie-aware Recall@K against all ids tied at the K-th boundary
* Classification accuracy and macro F1
* p50 / p95 / p99 latency over every stored observation
* Whether the planner used the HNSW index, and the applied session value

**Hypothesis:** Recall@K generally rises with `ef_search` while latency rises
too, with diminishing returns. It is **not** assumed to be monotonic for every
possible configuration: the graph search is heuristic, and a higher effort can
change which candidate list survives. A run that does not follow the expected
shape is a result, not an error. If the planner switches to a sequential scan,
the row is flagged and is not treated as an HNSW latency measurement.

### E3 — Dataset-size scaling

Repeat E1 and E2 with the training pool capped at several sizes (e.g. 10,000
and 50,000 rows) using stratified subsampling.

**Hypothesis:** exact-search latency grows roughly linearly with table size,
whereas HNSW latency grows more slowly. The relative speed-up should therefore
widen with size. This is a hypothesis about this implementation; the measured
ratios (13.1× at 9,999 rows and 48.8× at 50,000 rows, `ef = 10`) are reported
with their configuration, not generalised beyond it.

### E4 — Distance metric sensitivity

Repeat the comparison for `l2`, `cosine` and `l1`.

**Expected observation:** cosine distance is only meaningful as a similarity
measure when vector magnitude is irrelevant. The platform supports it, but it
does not L2-normalise vectors before a cosine run; feature standardisation is
not the same as unit-norm normalisation, and the two must not be described as
interchangeable.

### E5 — Neighbourhood size

Sweep `k` (e.g. 1, 3, 5, 7, 11, 21).

**Hypothesis:** larger `k` makes the majority vote more stable, so the
accuracy gap between exact and approximate retrieval narrows even when
Recall@K is unchanged. This has not been tested at scale and is future work.

## Metrics

| Metric | Definition |
| --- | --- |
| Accuracy | `correct / total` |
| Macro precision / recall / F1 | Unweighted mean over the 7 classes |
| Conventional Recall@K | Mean `|retrieved ∩ exact top-k ids| / k`, with exact ties broken by `ORDER BY distance, id` |
| Tie-aware Recall@K | Mean `|retrieved ∩ boundary candidates| / k`, capped at 1.0; the candidate set is every id with distance `<= boundary × (1 + 1e-6) + 1e-6` |
| Latency avg / p50 / p95 / p99 | Wall-clock time of a single retrieval query, over every stored observation |
| Index build time | Duration of a `CREATE INDEX ... USING hnsw` that actually ran; NULL when reused |
| Index reuse | Whether the index existed before the run, with the historical build time stored separately |
| Index size | `pg_relation_size` of the index in bytes |

Conventional Recall@K is not a mathematical lower bound on retrieval quality
in general. It is one only when every alternative at the boundary is equally
distant, all alternatives are in the candidate set, and the downstream task is
indifferent between them. Tie-aware Recall@K measures the second condition;
the first and third must be established rather than assumed.

## Duplicate vectors and ties

Duplicate feature vectors and tied boundary distances are measured per
dataset by `scripts/dataset_statistics.py`. The measurement covers unique
vector count, duplicate groups, largest group, mixed-label groups, tied
queries and label-changing ties. It is not inferred from the dataset's
reputation.

For the 50,000-row Covertype pool: 50,000 unique vectors, zero duplicate
groups, largest group 1, zero mixed-label groups, and zero tied queries in
1,000 (`k = 5`, L2). The raw 581,012-row file likewise contains 581,012
unique full-feature vectors. The earlier hypothesis that duplicate vectors
explain the Recall@K/accuracy gap is therefore not supported here. The
observed gap is more consistent with the robustness of a `k = 5` majority
vote, which is a hypothesis about the voting rule rather than a proven cause.

## Reproducing

```powershell
cd backend

# Seed a dataset
curl.exe -X POST http://127.0.0.1:8000/api/datasets/covertype `
  -H "Content-Type: application/json" `
  -d '{"name":"Forest CoverType","test_size":3000}'

# Measure duplicates and ties
.venv\Scripts\python -m scripts.dataset_statistics `
    --dataset-id 116 --k 5 --metric l2 --max-queries 1000 `
    --output ..\docs\results\duplicate_stats.json

# Run the matrix
.venv\Scripts\python -m scripts.run_experiments `
    --dataset-id 116 `
    --sizes 10000,50000 `
    --ef-search 10,25,50,100,200,400 `
    --k 5 --metric l2 --max-queries 1000 `
    --repetitions 3 --seed 0 `
    --output ..\docs\results\sweep.csv `
    --json ..\docs\results\sweep.json `
    --repetition-details ..\docs\results\repetition_details.csv `
    --repetition-summary ..\docs\results\repetition_summary.csv
```

## Threats to validity

* **Class imbalance.** CoverType is heavily imbalanced (in the 50,000-row
  pool, classes range from 236 to 24,380 rows). Accuracy therefore
  over-represents the majority classes; macro F1 is reported alongside it.
* **Correlated repetitions.** The passes share the query set, query order and
  index. They do not provide independent samples, and the reported intervals
  describe run-to-run variability only.
* **Deterministic quality metrics.** In this setup the neighbour sets are
  identical across passes, so recall and accuracy have zero between-pass
  variance. That is an observation about pgvector's deterministic search here,
  not a general property of HNSW.
* **Warm cache.** Measurements are taken after a warm-up query, so the
  PostgreSQL buffer cache is populated. Cold-cache behaviour is not measured.
* **Single machine, single dataset.** All timings come from one machine, and
  the full sweep uses one dataset. Absolute values are not a claim about
  production hardware or other data; only the relative difference between the
  retrieval strategies is claimed.
* **Small query sample.** 1,000 of 3,000 held-out rows are evaluated per run;
  0.1 pp corresponds to one query. Differences of that size are not
  interpreted as effects.
* **Query-per-request overhead.** Each query is issued separately so that a
  per-query latency can be measured; a batched workload would amortise the
  round-trip and change the speed-up ratio.
