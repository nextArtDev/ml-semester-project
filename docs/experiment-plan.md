# Experiment Plan

This document maps the research questions onto the experiments the platform
can run, states the measurement contract, and lists expected observations as
hypotheses rather than guaranteed outcomes.

## Datasets

| Dataset | Rows | Features | Classes | Source |
| --- | --- | --- | --- | --- |
| Forest CoverType | 581,012 | 54 | 7 | UCI ML Repository (seeded into the platform) |
| *Any uploaded CSV* | user-defined | user-defined | user-defined | CSV upload |

The reported sweep uses dataset 116: a 50,000-row training pool and a 3,000-row
held-out pool, fingerprint `6baa8fa283ea5a8267d3e964eedf5d2a` at full size.

## Preprocessing contract

1. **Stratified** train/test split, preserving class distribution.
2. Scaler and categorical mappings are fitted on the training split only. The
   `test_leakage_contract` test enforces this structurally.
3. The primary `covertype` mode standardises the 10 continuous features and
   keeps the 44 wilderness/soil indicators as 0/1. Arbitrary uploads use median
   imputation, one-hot encoding of nominal columns and standardisation of the
   numeric block. `standardize_all` is an explicit alternative, not the primary
   setup.
4. The fitted state and a dataset fingerprint (row count, ordered ids/labels,
   vector-content checksum) are stored on the dataset row and copied to each
   experiment.

## Measurement contract

Five quantities are reported separately:

| Quantity | Definition | Example |
| --- | --- | --- |
| Held-out test pool | Size of the query pool | 3,000 |
| Unique evaluated queries | Queries actually executed | 1,000 |
| Repetitions | Complete timed passes over that query set | 3 |
| Benchmark rounds | Permutations of the configuration matrix | 3 |
| Latency observations | Stored per-query samples | 42 × 3,000 = 126,000 |

`--repetitions N` executes N full passes; it never multiplies a counter. Every
sample carries `repetition` and `query_index`, and the stored count is verified
against `repetitions × unique_queries`. One warm-up query runs before the first
timed query and is never stored. The query order is identical across
repetitions, so repetitions measure run-to-run variability, not population
sampling.

`--rounds R` runs the matrix R times under a seeded balanced order. Each round
is a permutation of all configurations; round *r* rotates a seeded base order
by `floor(n_configurations / R)` positions. Three rounds guarantee that every
configuration appears both early and late. The actual order, round, position
and seed are persisted and exported. The HNSW index is not rebuilt between
`ef_search` values or rounds. All measurements are warm-cache; no cold-cache
procedure is executed or claimed.

## Query selection

The default is a seeded stratified sample of the held-out pool: proportional
largest-remainder allocation across classes, with at least one slot per
present class when the requested sample is large enough, then seeded sampling
within each class. Selected ids are sorted, identical for exact and HNSW,
stored as `evaluated_query_ids` and exported to `query_samples.json`. The pool
and evaluated class distributions are persisted on every experiment row.
`--max-queries 0` evaluates the entire pool; `first` mode is retained only for
reproducing older runs and is never described as representative.

## Timing definitions

All phases use `time.perf_counter()` and are non-overlapping:

| Field | Includes |
| --- | --- |
| `total_experiment_runtime_s` | Whole `run_experiment` call through persistence |
| `index_preparation_s` | Build-or-reuse decision, `ANALYZE`, metadata readback |
| `index_build_s` | Only a fresh `CREATE INDEX`; 0 when reused, subset of preparation |
| `ground_truth_computation_s` | Exhaustive query and boundary candidates |
| `retrieval_total_s` | Sum of individually stored query latencies |
| `metrics_computation_s` | Voting, scores, cross-repetition summary |
| `persistence_s` | Experiment, repetition and latency inserts |
| `orchestration_overhead_s` | Residual: loading, counting, fingerprint, loop overhead |

The retired `api_overhead_s` (runtime minus query time) was not API overhead:
it included ground-truth computation and almost every other phase. HTTP request
latency, if needed, must be measured at the HTTP boundary.

## Experiments

### E1 — Exact baseline

Establishes the reference neighbours, accuracy and latency for a dataset, `k`
and metric. It uses the pgvector distance *function*, so the plan is an
exhaustive sequential scan by construction, and orders by `(distance, id)` so
repeated runs return the same ids when distances tie.

### E2 — `ef_search` sweep

For fixed dataset, `k`, metric and pool, run HNSW at several `ef_search`
values and record conventional and tie-aware Recall@K, accuracy, macro F1,
latency over all observations, the applied session parameter and the verified
access path. HNSW keeps only the index-compatible distance ordering; an id
tiebreak there would make the planner abandon the index.

**Hypothesis:** recall generally rises with effort while latency rises too,
with diminishing returns. It is **not** assumed to be monotonic: graph
traversal is heuristic, and higher effort can change which candidate list
survives. A non-monotonic result is a result, not an error.

### E3 — Dataset-size scaling

Repeat E1 and E2 with training pools capped at several sizes using stratified
subsampling. **Hypothesis:** exact latency grows roughly linearly while HNSW
grows more slowly, so the relative speed-up widens with size. This is a
hypothesis about this implementation, not a general law.

### E4 — Distance metric sensitivity

Repeat the comparison for `l2`, `cosine` and `l1`. Cosine distance is only a
similarity measure when magnitude is irrelevant; vectors are not L2-normalised
before a cosine run, and feature standardisation is not unit-norm
normalisation.

### E5 — Neighbourhood size

Sweep `k` (e.g. 1, 3, 5, 7, 11, 21). **Hypothesis:** larger `k` makes the vote
more stable, narrowing the accuracy gap even when Recall@K is unchanged. Not
tested at scale; future work.

## Metrics

| Metric | Definition |
| --- | --- |
| Accuracy | Correct / total |
| Macro precision / recall / F1 | Unweighted mean over the 7 classes |
| Conventional Recall@K | Mean `|retrieved ∩ exact top-k ids| / k`, exact ties broken by `ORDER BY distance, id` |
| Tie-aware Recall@K | Mean `|retrieved ∩ boundary candidates| / k`, capped at 1.0; boundary = `distance ≤ b × (1 + rel) + abs` |
| Latency avg / p50 / p95 / p99 | Wall-clock time of one retrieval query over all stored observations |
| Index build / reuse | Fresh build duration only; historical time stored separately on reuse |
| Index size | `pg_relation_size` of the live index |

The default tolerance is `rel = 1e-6`, `abs = 1e-6`. It is validated for L2,
cosine and L1 with synthetic cases: exact ties, near-boundary differences,
large boundaries, tiny boundaries, and more than K equally valid candidates.
It is recorded per experiment. Conventional Recall@K is not a mathematical
lower bound on retrieval quality in general; that would require every
alternative at the boundary to be equally distant, all alternatives to be in
the candidate set, and the task to be indifferent between them.

## Duplicate vectors and ties

Measured, never assumed, by `scripts/dataset_statistics.py`. For the 50,000-row
Covertype pool: 50,000 unique vectors, zero duplicate groups, zero mixed-label
groups, zero tied queries in 1,000 (`k = 5`, L2), largest boundary candidate
set 5. The raw 581,012-row file has 581,012 unique full-feature vectors. The
duplicate explanation for the recall/accuracy gap is therefore rejected on
this dataset. See [`results/duplicate_stats.json`](results/duplicate_stats.json).

## Reproducing

```powershell
cd backend

# Duplicate/tie measurement
.venv\Scripts\python -m scripts.dataset_statistics `
    --dataset-id 116 --k 5 --metric l2 --max-queries 1000 `
    --output ..\docs\results\duplicate_stats.json

# Full balanced benchmark
.venv\Scripts\python -m scripts.run_experiments `
    --dataset-id 116 --sizes 10000,50000 `
    --ef-search 10,25,50,100,200,400 --k 5 --metric l2 `
    --max-queries 1000 --repetitions 3 --seed 0 `
    --rounds 3 --order-seed 0 `
    --query-sampling stratified --query-sample-seed 0 `
    --output ..\docs\results\sweep.csv `
    --json ..\docs\results\sweep.json `
    --repetition-details ..\docs\results\repetition_details.csv `
    --repetition-summary ..\docs\results\repetition_summary.csv `
    --query-samples ..\docs\results\query_samples.json
```

## Threats to validity

* **Query sampling.** 1,000 of 3,000 held-out rows are evaluated; a 0.1 pp
  difference is one query. The stratified sample is close to the pool
  distribution but is not the pool.
* **Class imbalance.** Classes range from 14 to 1,463 rows in the 3,000-row
  held-out pool. Macro F1 accompanies accuracy.
* **Correlated repetitions and rounds.** Repetitions share queries, order and
  index. Rounds change configuration order only. Neither provides independent
  samples of the population.
* **Cache and host effects.** Warm buffer cache, one machine, localhost,
  per-query round-trip. Cold-cache behaviour and batched throughput are not
  measured.
* **Single dataset and metric in the full sweep.** The speed-up numbers are
  specific to Covertype, L2 and this machine; no generalisation is claimed.
* **Deterministic quality in this setup.** Quality metrics were identical
  across passes with a fixed graph and query order. That is an observation
  about this pgvector build, not a general property of HNSW.
