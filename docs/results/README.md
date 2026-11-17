# Experimental Results

Artifacts produced by `backend/scripts/run_experiments.py` and
`backend/scripts/dataset_statistics.py`:

| File | Contents |
| --- | --- |
| [`sweep.csv`](sweep.csv) | One row per experiment (52 columns): configuration, quality, latency, index metadata and environment |
| [`sweep.json`](sweep.json) | The identical rows as JSON, for programmatic use |
| [`repetition_details.csv`](repetition_details.csv) | One row per experiment **and repetition** (42 rows): that pass's latency distribution, Recall@K, tie-aware Recall@K and classification metrics |
| [`repetition_summary.csv`](repetition_summary.csv) | Cross-repetition median, standard deviation, min/max and 95 % interval for each metric (140 rows = 14 experiments × 10 metrics) |
| [`duplicate_stats.json`](duplicate_stats.json) | Measured duplicate-vector and tied-boundary statistics for the 50,000-row training pool |

Every row was produced by the same `run_experiment` service the web API uses.
The raw CSVs are the primary record; the numbers quoted below are read from
them.

## Setup

| Parameter | Value |
| --- | --- |
| Dataset | UCI Forest CoverType, ingested as dataset 116 |
| Preprocessing | `covertype`: 10 continuous features standardised on the training split, 44 binary indicators kept as 0/1 |
| Dimension | 54 |
| Split | stratified, `seed = 42`, 50,000 train / 3,000 held-out |
| Training pools | 9,999 and 50,000 rows (stratified subsample of the 50,000-row pool, `seed = 0`) |
| Dataset fingerprints | 9,999 rows: `af656f7bffa604fefe323bd8f6d929d1`; 50,000 rows: `6baa8fa283ea5a8267d3e964eedf5d2a` |
| `k` | 5 |
| Distance metric | L2 |
| HNSW build | `m = 16`, `ef_construction = 64` |
| Search effort | `ef_search` ∈ {10, 25, 50, 100, 200, 400} |
| Unique queries per run | 1,000 |
| Repetitions | 3 complete passes, in identical query order |
| Latency observations | 1,000 × 3 = 3,000 per run |
| Warm-up | 1 discarded query per batch, never stored |
| Tie tolerance | relative `1e-6`, absolute `1e-6` |
| Environment | Python 3.12.1, PostgreSQL 18.0, pgvector 0.8.7, 12 CPU cores, Windows 11, single machine, warm buffer cache |

### What is counted, and what is not

Four different quantities appear in this experiment and must not be confused:

* **Held-out test pool**: 3,000 rows. This is the population from which
  queries are drawn.
* **Unique evaluated queries**: 1,000 per run. The first 1,000 test rows by
  `id`, the same set for the exact and every HNSW run at a given pool size.
* **Repetitions**: 3. Each repetition is a complete timed pass over all 1,000
  queries. The query order is identical in every pass, so repetitions expose
  run-to-run timing variability; they are **not** independent samples of a
  population and are not a substitute for evaluating more queries.
* **Latency observations**: 3,000 per run. Each individual observation is
  stored in `query_latencies` with its repetition number and query index.

Across the 14 experiments the database holds 42,000 latency rows and 42
per-repetition rows, which is exactly `14 × 3,000` and `14 × 3`.

## Raw measurements

Recall@K and tie-aware Recall@K are identical in every row because the
measured boundary-tie count is zero (see below); both are reported anyway,
because that agreement is a result, not an assumption.

| Rows | Method | `ef` | Recall@5 | Tie-aware | Accuracy | Macro P | Macro R | Macro F1 | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Speed-up (p50) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9,999 | exact | — | 1.0000 | 1.0000 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 2.839 | 2.769 | 3.278 | 3.728 | — |
| 9,999 | hnsw | 10 | 0.9952 | 0.9952 | 0.7760 | 0.6316 | 0.4906 | 0.5070 | 0.218 | 0.211 | 0.267 | 0.331 | 13.1× |
| 9,999 | hnsw | 25 | 0.9996 | 0.9996 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 0.264 | 0.246 | 0.312 | 0.383 | 11.3× |
| 9,999 | hnsw | 50 | 0.9998 | 0.9998 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 0.305 | 0.297 | 0.377 | 0.443 | 9.3× |
| 9,999 | hnsw | 100 | 1.0000 | 1.0000 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 0.396 | 0.387 | 0.483 | 0.555 | 7.2× |
| 9,999 | hnsw | 200 | 1.0000 | 1.0000 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 0.584 | 0.564 | 0.696 | 0.786 | 4.9× |
| 9,999 | hnsw | 400 | 1.0000 | 1.0000 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 0.925 | 0.908 | 1.111 | 1.266 | 3.0× |
| 50,000 | exact | — | 1.0000 | 1.0000 | 0.8560 | 0.7979 | 0.7021 | 0.7314 | 11.262 | 11.226 | 11.964 | 12.596 | — |
| 50,000 | hnsw | 10 | 0.9822 | 0.9822 | 0.8550 | 0.7978 | 0.7013 | 0.7309 | 0.247 | 0.230 | 0.295 | 0.358 | 48.8× |
| 50,000 | hnsw | 25 | 0.9926 | 0.9926 | 0.8580 | 0.8031 | 0.7099 | 0.7403 | 0.278 | 0.269 | 0.349 | 0.413 | 41.7× |
| 50,000 | hnsw | 50 | 0.9978 | 0.9978 | 0.8570 | 0.8026 | 0.7097 | 0.7400 | 0.339 | 0.331 | 0.431 | 0.490 | 33.9× |
| 50,000 | hnsw | 100 | 0.9998 | 0.9998 | 0.8570 | 0.8026 | 0.7097 | 0.7400 | 0.465 | 0.442 | 0.590 | 0.763 | 25.4× |
| 50,000 | hnsw | 200 | 0.9998 | 0.9998 | 0.8570 | 0.8026 | 0.7097 | 0.7400 | 0.670 | 0.655 | 0.869 | 0.993 | 17.1× |
| 50,000 | hnsw | 400 | 0.9998 | 0.9998 | 0.8570 | 0.8026 | 0.7097 | 0.7400 | 1.095 | 1.079 | 1.378 | 1.528 | 10.4× |

Every HNSW row reports `used_index_scan = True`, and the applied
`hnsw.ef_search` read back from the server equals the requested value. Every
exact row reports `used_sequential_scan = True`; the baseline is exhaustive by
construction because it uses the distance *function*.

Index metadata, taken from the live index rather than from Python defaults:

| Training rows | `m` | `ef_construction` | Fresh build | Index size | Reused by later `ef` values |
| --- | --- | --- | --- | --- | --- |
| 9,999 | 16 | 64 | 597.6 ms | 5,234,688 B (4.99 MiB) | yes — `index_build_ms` is NULL, `index_reused = True` |
| 50,000 | 16 | 64 | 3,239.0 ms | 26,107,904 B (24.90 MiB) | yes — `index_build_ms` is NULL, `index_reused = True` |

A reused index's historical build time is stored in
`index_reused_build_ms`, never in `index_build_ms`, so a low-`ef` row is not
misread as having paid a multi-second build cost.

## Repetition-level results

The three passes executed the same 1,000 queries in the same order against the
same index. Two different things are therefore true, and both are visible in
[`repetition_details.csv`](repetition_details.csv):

1. **Quality metrics are deterministic across passes.** Recall@K, tie-aware
   Recall@K, accuracy, macro precision/recall/F1 and the confusion matrix are
   identical in repetition 1, 2 and 3 for every configuration (standard
   deviation 0). The same neighbour sets are returned because neither the
   query order nor the index changes between passes. Repetitions do not create
   independent estimates of recall or accuracy on this workload.
2. **Latency varies between passes.** The per-repetition mean latency
   standard deviation ranges from 0.0013 ms to 0.20 ms. The largest *relative*
   dispersion is 6.9 % (9,999 rows, `ef = 25`, mean 0.264 ms, SD 0.018 ms);
   most configurations are below 3 %. With only three repetitions the 95 %
   interval is wide — for example, 50,000 rows at `ef = 100` has a
   per-repetition mean of 0.465 ms with a 95 % interval of
   [0.415, 0.516] ms — and the interval is computed with `t(2) = 4.303`.

`repetition_summary.csv` reports the median, SD, min, max and 95 % interval for
each metric. Those intervals describe **run-to-run variability of this
workload**, not sampling error over the 3,000 test rows, because the three
passes are correlated by construction. Percentiles in `sweep.csv` are computed
over all 3,000 individual observations; they are not averages of three
per-pass percentiles.

## How Recall@K is defined

Both metrics use the same exhaustive reference, computed with the pgvector
distance *function*:

* **Conventional Recall@K**: `|retrieved ∩ exact top-k ids| / k`, averaged
  over queries. The exact top-k is made deterministic with
  `ORDER BY distance, id`, so the reference does not depend on the executor's
  arbitrary choice among tied rows.
* **Tie-aware Recall@K**: the candidate set is every training id whose
  distance is `<= boundary × (1 + 1e-6) + 1e-6`, where `boundary` is the exact
  K-th neighbour distance. The score is
  `|retrieved ∩ candidate set| / k`, capped at 1.0. Returning a
  boundary-tied row instead of the reference row therefore counts as valid.
  Dividing by `k` keeps the metric comparable with the conventional one.

The tolerance absorbs float4/float8 representation error. It is persisted per
experiment as `tie_tolerance_rel` / `tie_tolerance_abs`; it does not define
"equal" loosely enough to admit genuinely different distances.

Conventional Recall@K is **not** a mathematical lower bound on retrieval
quality in general. It becomes one only under conditions that must be
established, not assumed: every alternative returned neighbour at the boundary
is equally distant, all such alternatives are present in the reference
candidate set, and the task is indifferent between them. The tie-aware metric
makes the second condition explicit; the third is a task-level judgement.

## Duplicate vectors and ties in Covertype

Measured, not assumed
([`duplicate_stats.json`](duplicate_stats.json), 50,000-row pool, `k = 5`,
L2, 1,000 queries):

| Statistic | Value |
| --- | --- |
| Training vectors | 50,000 |
| Unique feature vectors | 50,000 |
| Duplicate groups | 0 |
| Duplicate rows | 0 (0.0 %) |
| Largest duplicate group | 1 |
| Groups containing more than one label | 0 |
| Queries with a tied exact K-th distance | 0 / 1,000 |
| Largest candidate set at the boundary | 5 (= `k`) |
| Ties whose alternative label would change the vote | 0 |

The raw UCI file was also checked directly over all 581,012 rows: 581,012
unique full-feature vectors, zero duplicate groups. The earlier caveat that
duplicate Covertype vectors make Recall@K a lower bound is therefore **not
supported** for this dataset. The conventional and tie-aware metrics coincide
here because there are no boundary ties, and the reported Recall@K is
identifier-level agreement with an untied reference.

This does not mean ties can never occur on other data, nor that distinct
vectors cannot share a distance (for example, symmetric points around a
query). It means the hypothesis that duplicates explain the observed
recall/accuracy gap is rejected for this pool by measurement.

## Findings

### 1. `ef_search` controls a recall/latency trade-off, but the curve is compressed

At 50,000 rows, Recall@5 rises from 0.9822 at `ef = 10` to 0.9978 at `ef = 50`
and 0.9998 at `ef = 100`, then stays at 0.9998 through `ef = 400`. Median
latency rises from 0.230 ms to 1.079 ms over the same range. At 9,999 rows the
curve starts higher: 0.9952 at `ef = 10`, reaching 1.0000 at `ef = 100`.

In this run recall increased with `ef_search` at both sizes, but that is an
observation about these six configurations, not a guarantee. Recall is not
required to be monotonic for every possible `ef_search`, dataset or index
build: graph traversal is heuristic, and a higher effort can change which
candidate list survives without strictly containing the lower-effort result.
The plateau at 0.9998 rather than 1.0000 at 50,000 rows is a concrete reminder
that near-perfect is not perfect.

The curve is flatter than on a purely continuous feature space. One
contributing factor is the representation: 44 of the 54 coordinates are
forest-cover indicators, so many rows share coordinate values even though no
two full feature vectors are identical. That is a hypothesis about graph
locality for this representation, not a demonstrated cause; the measured
duplicate count rules out exact duplicate vectors as the explanation.

### 2. Classification quality moves much less than Recall@K

At 50,000 rows and `ef = 10`, HNSW missed 1.78 percentage points of exact
neighbours (Recall@5 0.9822 versus 1.0000), while accuracy moved from 0.8560
to 0.8550 (0.10 pp) and macro F1 from 0.7314 to 0.7309 (0.05 pp). At
`ef = 25` and above, accuracy is 0.8570–0.8580 and macro F1 0.7400–0.7403,
slightly *above* the exact baseline.

At 9,999 rows and `ef = 10`, accuracy is 0.7760 versus 0.7750 and macro F1
0.5070 versus 0.5069 — again slightly higher, not lower. With 1,000 queries a
difference of 0.1 pp is one query, so these small positive differences should
not be read as evidence that approximation improves classification. The
defensible statement is narrower: the classification difference is at most
0.2 pp in either direction in this sweep, while the retrieval gap reaches
1.78 pp.

The mechanism is the `k = 5` majority vote: a missed neighbour changes the
prediction only if it flips the vote. This makes classification robust to
neighbour-set error on this workload. It also means recall and accuracy must
be reported separately; neither is a proxy for the other.

### 3. The speed-up grows with the training pool

At `ef = 10` the exact/HNSW median-latency ratio is 13.1× for 9,999 rows and
48.8× for 50,000 rows. At `ef = 400` it is 3.0× and 10.4×. Exact p50 grows from
2.769 ms to 11.226 ms (4.1× for 5.0× more rows, close to the expected linear
scan), while HNSW p50 at `ef = 10` grows only from 0.211 ms to 0.230 ms.

The advantage is therefore not a constant factor, and it shrinks as the search
effort rises. Both directions of that statement matter.

### 4. Build and storage cost are one-off

`CREATE INDEX ... USING hnsw` took 597.6 ms for 9,999 rows and 3,239.0 ms for
50,000 rows, producing a 4.99 MiB and a 24.90 MiB index. Each index was built
once and reused for the remaining five `ef_search` values: the later rows
record `index_reused = True`, `index_build_ms = NULL`, and the historical
build time separately.

An index is not free even when reused: it occupies storage, consumes build
time on the first run, and must be rebuilt when the metric or build parameters
change.

### 5. Query time is a small part of end-to-end run time

`api_overhead_s` (run wall-clock minus the sum of measured query latencies)
ranges from 4.7 s to 21.3 s per experiment. That time covers ground-truth
computation, classification and persistence — not retrieval. The speed-up
figures above refer to query latency only; conflating the two would understate
the retrieval difference and overstate what a deployed API would observe.

## What changed since the earlier results

The figures previously published in this folder (a 45× speed-up and 86.6 %
recall) came from a pipeline that was **not measuring what it claimed**:

| Defect | Effect |
| --- | --- |
| `SET LOCAL hnsw.ef_search` on an autocommit pool | Silently discarded — every run used pgvector's default of 40, so the "sweep" was flat |
| Nominal/categorical encoding was ordinal | Distances between categories were arbitrary |
| Repetition counters were not backed by repeated passes | The reported "3 repetitions" could be a multiplied query count |
| No execution-plan check | A sequential-scan fallback would have been reported as HNSW latency |
| `stratified_split(train_fraction=…)` treated the argument as a test fraction | Splits were not the requested size |
| Index build time reported on reused-index rows | A historical build was presented as a fresh cost |

All six are fixed and covered by tests; the artifacts above were regenerated
from the corrected pipeline. The old figures should not be quoted.

## Caveats

* **Correlated repetitions.** The three passes reuse the same query order,
  training pool and index. Their spread measures run-to-run variability, not
  population sampling; do not treat 3,000 latency observations as 3,000
  independent experiments.
* **One dataset, one machine.** The qualitative findings are hypotheses beyond
  this workload. Absolute latencies describe localhost with a warm buffer
  cache, not a production deployment.
* **Deterministic HNSW in this setup.** The quality metrics did not vary across
  passes. That is a property of this pgvector build with a fixed graph and
  query order, not a claim that HNSW is deterministic in general.
* **Small query sample per run.** 1,000 of 3,000 held-out rows are evaluated.
  Differences of 0.1 pp correspond to one query.
* **Class imbalance.** Covertype's classes range from 236 to 24,380 rows in
  the 50,000-row pool, which is why macro F1 accompanies accuracy.
* **Per-query round-trip.** Queries are issued one at a time so that a
  per-query latency exists; a batched workload would amortise the round-trip.

See [`../experiment-plan.md`](../experiment-plan.md) for the full list of
threats to validity.
