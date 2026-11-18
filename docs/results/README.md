# Experimental Results

Raw artefacts produced by `backend/scripts/run_experiments.py` and
`backend/scripts/dataset_statistics.py`:

| File | Contents |
| --- | --- |
| [`sweep.csv`](sweep.csv) | One row per experiment (64 columns): configuration, quality, latency, timing decomposition, sampling and execution-order metadata |
| [`sweep.json`](sweep.json) | The identical rows as JSON |
| [`repetition_details.csv`](repetition_details.csv) | One row per experiment **and repetition** (126 rows) |
| [`repetition_summary.csv`](repetition_summary.csv) | Median, SD, min/max and 95 % interval per metric (420 rows = 42 × 10) |
| [`query_samples.json`](query_samples.json) | The exact held-out ids evaluated by each experiment |
| [`duplicate_stats.json`](duplicate_stats.json) | Measured duplicate-vector and boundary-tie statistics |
| [`v1/`](v1/README.md) | The **unmodified** previous artifacts, preserved for comparison |

Every row was produced by the same `run_experiment` service the web API uses.
The CSVs are the primary record; every number below was read back from the
artefacts and cross-checked against the database (0 mismatches).

## Setup

| Parameter | Value |
| --- | --- |
| Dataset | UCI Forest CoverType, ingested as dataset 116 |
| Preprocessing | `covertype`: 10 continuous features standardised on the training split, 44 binary indicators kept as 0/1 |
| Dimension | 54 |
| Split | stratified, `seed = 42`, 50,000 train / 3,000 held-out |
| Training pools | 9,999 and 50,000 rows (stratified subsample, `seed = 0`) |
| Dataset fingerprints | 9,999 rows: `af656f7bffa604fefe323bd8f6d929d1`; 50,000 rows: `6baa8fa283ea5a8267d3e964eedf5d2a` |
| `k` | 5 |
| Distance metric | L2 |
| HNSW build | `m = 16`, `ef_construction = 64`; built once per pool, reused thereafter |
| Search effort | `ef_search` ∈ {10, 25, 50, 100, 200, 400} |
| Query selection | seeded **stratified** sample of 1,000 of 3,000 held-out rows, `query_sample_seed = 0` |
| Repetitions | 3 complete passes over the same query set, in the same order |
| Benchmark rounds | 3 paired rounds per pool size, `order_seed = 0` |
| Latency observations | 1,000 × 3 × 3 = 9,000 per (size, configuration) |
| Warm-up | 1 discarded query per batch, never stored |
| Cache | warm PostgreSQL buffer cache; **no cold-cache procedure was executed** |
| Tie tolerance | relative `1e-6`, absolute `1e-6` |
| Environment | Python 3.12.1, PostgreSQL 18.0, pgvector 0.8.7, 12 CPU cores, Windows 11 |

### What is counted

| Quantity | Value |
| --- | --- |
| Held-out test pool | 3,000 rows per size |
| Unique evaluated queries | 1,000 per experiment |
| Repetitions | 3 |
| Benchmark rounds | 3 |
| Experiments | 42 (2 sizes × 7 configurations × 3 rounds) |
| Latency observations | 126,000 = 42 × 3,000 |
| Per-repetition rows | 126 = 42 × 3 |

Repetitions share the query set, the query order and the index. They measure
**run-to-run timing variability**, not sampling error over the held-out pool.
Rounds reshuffle the *configuration* order; they do not resample queries.

## Query sampling and class distribution

The first 1,000 test ids (used in v1) were reproducible but not justified as a
representative sample. This run uses the seeded stratified selector: allocate
the 1,000 slots proportionally to class frequency with a largest-remainder
rule, then sample within each class with a seed derived from
`query_sample_seed` and the class label. The selected ids are sorted and
identical for the exact and every HNSW run.

| Label | Test-pool rows | Pool share | Evaluated rows | Evaluated share |
| --- | --- | --- | --- | --- |
| 0 | 1,094 | 36.47 % | 364 | 36.4 % |
| 1 | 1,463 | 48.77 % | 490 | 49.0 % |
| 2 | 184 | 6.13 % | 61 | 6.1 % |
| 3 | 14 | 0.47 % | 4 | 0.4 % |
| 4 | 49 | 1.63 % | 16 | 1.6 % |
| 5 | 90 | 3.00 % | 30 | 3.0 % |
| 6 | 106 | 3.53 % | 35 | 3.5 % |

Every class is represented; the largest absolute deviation in share is 0.23
percentage points. The exact ids for each experiment are in
`query_samples.json`, and the distributions are persisted on each experiment
row. This is a *sample*, not the full pool: 1,000 of 3,000 rows were evaluated
because the 42-run balanced design would otherwise triple the cost of a
full-pool sweep.

## Execution-order design

Within each pool size every round is a permutation of the seven configurations
(exact plus six `ef_search` values). The base permutation comes from
`random.Random(order_seed)`, and round *r* rotates it by
`floor(n_configurations / n_rounds)` positions. Three rounds are used because
they guarantee that every configuration appears in both the early and the late
part of the sequence. Actual positions:

| Configuration | Round 1 | Round 2 | Round 3 |
| --- | --- | --- | --- |
| exact | 4 | 2 | 7 |
| HNSW ef=10 | 3 | 1 | 6 |
| HNSW ef=25 | 2 | 7 | 5 |
| HNSW ef=50 | 6 | 4 | 2 |
| HNSW ef=100 | 1 | 6 | 4 |
| HNSW ef=200 | 5 | 3 | 1 |
| HNSW ef=400 | 7 | 5 | 3 |

The order, seed, round and position are persisted on every experiment row and
exported in `sweep.csv`. The HNSW index is built once per pool and reused by
all 21 runs of that pool; it is **not** rebuilt between `ef_search` values or
rounds. Speed-ups are computed against the mean exact baseline over the three
rounds, so a configuration's ratio does not depend on which run happened to
precede it.

## Results

Each cell is the mean across the three balanced rounds. `p50` is computed over
the 3,000 individual latency observations of that run; the ± value is the SD of
the three per-round p50 figures, i.e. run-to-run variability.

| Rows | Method | `ef` | Recall@5 | Tie-aware | Accuracy | Macro F1 | p50 (ms) | Speed-up (p50) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9,999 | exact | — | 1.0000 | 1.0000 | 0.7530 | 0.5255 | 2.765 ± 0.013 | — |
| 9,999 | hnsw | 10 | 0.9938 | 0.9938 | 0.7530 | 0.5241 | 0.214 ± 0.001 | 12.9× |
| 9,999 | hnsw | 25 | 0.9988 | 0.9988 | 0.7530 | 0.5255 | 0.250 ± 0.003 | 11.1× |
| 9,999 | hnsw | 50 | 0.9994 | 0.9994 | 0.7530 | 0.5255 | 0.304 ± 0.003 | 9.1× |
| 9,999 | hnsw | 100 | 1.0000 | 1.0000 | 0.7530 | 0.5255 | 0.399 ± 0.003 | 6.9× |
| 9,999 | hnsw | 200 | 1.0000 | 1.0000 | 0.7530 | 0.5255 | 0.576 ± 0.001 | 4.8× |
| 9,999 | hnsw | 400 | 1.0000 | 1.0000 | 0.7530 | 0.5255 | 0.929 ± 0.014 | 3.0× |
| 50,000 | exact | — | 1.0000 | 1.0000 | 0.8530 | 0.7166 | 10.812 ± 0.013 | — |
| 50,000 | hnsw | 10 | 0.9902 | 0.9902 | 0.8520 | 0.7184 | 0.240 ± 0.015 | 45.2× |
| 50,000 | hnsw | 25 | 0.9966 | 0.9966 | 0.8520 | 0.7162 | 0.274 ± 0.006 | 39.5× |
| 50,000 | hnsw | 50 | 0.9978 | 0.9978 | 0.8520 | 0.7162 | 0.337 ± 0.004 | 32.1× |
| 50,000 | hnsw | 100 | 0.9998 | 0.9998 | 0.8520 | 0.7162 | 0.461 ± 0.017 | 23.5× |
| 50,000 | hnsw | 200 | 0.9998 | 0.9998 | 0.8520 | 0.7162 | 0.683 ± 0.003 | 15.8× |
| 50,000 | hnsw | 400 | 0.9998 | 0.9998 | 0.8520 | 0.7162 | 1.147 ± 0.005 | 9.4× |

Every HNSW row reports `used_index_scan = True` and the applied session
`hnsw.ef_search` equals the requested value (36/36). Every exact row reports a
sequential scan (6/6). The exact query now orders by `(distance, id)`, so
repeated exact retrieval returns the same neighbour ids when distances tie;
the HNSW query keeps only the index-compatible distance ordering.

Conventional and tie-aware Recall@K coincide in every row because the measured
boundary-tie count on this pool is zero (`duplicate_stats.json`). That
agreement is a measurement, not an assumption.

### Index build and reuse

| Training rows | Fresh build | Index size | Reused runs | Historical build stored separately |
| --- | --- | --- | --- | --- |
| 9,999 | 574.0 ms | 5,234,688 B | 20 | yes, in `index_reused_build_ms` |
| 50,000 | 3,231.1 ms | 26,107,904 B | 20 | yes, in `index_reused_build_ms` |

Only 2 of 42 rows report a fresh build. Every reused row has
`index_build_ms = NULL`; the historical cost is never presented as a cost of
that configuration.

### Timing decomposition

Mean seconds across the 42 runs. These are non-overlapping intervals measured
with `time.perf_counter()`; the sum is at most the total runtime.

| Phase | Mean (s) | What it includes |
| --- | --- | --- |
| `total_experiment_runtime_s` | 16.155 | Whole run, start through persistence |
| `index_preparation_s` | 0.115 | Build-or-reuse decision, `ANALYZE`, metadata readback |
| `index_build_s` | 0.091 | Subset of the above: only a fresh `CREATE INDEX`; 0 on reuse |
| `ground_truth_computation_s` | 10.689 | Exhaustive exact query + boundary candidates for all 1,000 queries |
| `retrieval_total_s` | 4.203 | Sum of the individually timed method queries |
| `metrics_computation_s` | 0.019 | Majority vote, accuracy/F1/recall, cross-repetition summary |
| `persistence_s` | 0.107 | Experiment, per-repetition and latency inserts |
| `orchestration_overhead_s` | 1.022 | Residual: dataset/query loading, counting, fingerprint, loop overhead |

The old `api_overhead_s` (`total_runtime − query_time`) would have reported
roughly 12 s here and called it "API overhead", although almost all of it was
ground-truth computation. The new fields separate those costs. Retrieval is a
minority of run time in this harness; the reported speed-ups compare query
latency only.

## Interpretation: what each effect can and cannot explain

* **Query sampling.** The stratified sample changed the measured accuracy from
  0.7750 (v1, first 1,000 ids) to 0.7530 at 9,999 rows and from 0.8560 to
  0.8530 at 50,000 rows. That is a property of the *query set*, not a
  regression in the method: both arms use the same ids. The class distribution
  is close to the pool, but 1,000 rows still cannot resolve 0.1 pp differences.
* **Cache and execution order.** All runs are warm-cache. Balanced rounds
  remove the fixed exact-first / ascending-`ef` order, so no configuration is
  systematically advantaged by running first. This does not remove warm-cache
  effects or measure cold starts.
* **Run-to-run timing variability.** The SD of the per-round p50 ranges from
  0.001 ms to 0.017 ms. The three passes are correlated by construction; their
  spread is not a confidence interval for the held-out population.
* **Retrieval approximation quality.** Recall@5 ranges from 0.9902 to 1.0000
  across tested configurations. It rose with `ef_search` in this run, but the
  curve is not guaranteed to be monotonic for other data, metrics or builds.
* **Classification quality.** Accuracy differs from the exact baseline by at
  most 0.1 pp here. A missed neighbour changes the vote only when it flips the
  majority, which is why classification moves less than recall on this
  workload. This is a hypothesis about `k = 5` voting, not a general law.
* **Index construction and storage.** One build per pool costs 0.57 s / 3.23 s
  and 4.99 MiB / 24.90 MiB. Reuse makes later runs cheap, but the index still
  consumes storage and must be rebuilt when the metric or build parameters
  change.

## Differences from v1

| Aspect | v1 | Current |
| --- | --- | --- |
| Configuration order | fixed per size | 3 seeded balanced rounds |
| Query selection | first 1,000 ids | seeded stratified sample of 1,000 of 3,000 |
| Rounds | 1 | 3 |
| Latency observations per configuration | 3,000 | 9,000 |
| Timing | `api_overhead_s` residual | explicit phase decomposition |
| Exact ordering | `ORDER BY distance` | `ORDER BY distance, id` |
| Tests | 164 | 211 passed, 1 documented skip |

The v1 numbers must not be mixed with the current numbers in one figure or
table. Both sets are preserved; the current artefacts are the ones to quote.

## Caveats

* One dataset, one machine, warm cache, per-query round-trip, 1,000 of 3,000
  held-out rows, class imbalance, three correlated repetitions, and a single
  L2 metric in the full sweep. None of these findings is claimed to generalise
  to every dataset.
* The `k`, metric and index-build-parameter sweeps remain future work.
* Conventional Recall@K is not a mathematical lower bound on retrieval
  quality in general; see [`../experiment-plan.md`](../experiment-plan.md).
