# Experimental Results

Raw artefacts produced by `backend/scripts/run_experiments.py`:

* [`sweep.csv`](sweep.csv) — one row per experiment (38 columns)
* [`sweep.json`](sweep.json) — the same data as JSON

Every row was produced by the same `run_experiment` service the web API uses,
with `repetitions = 3`, and every row was verified against
`EXPLAIN (ANALYZE, BUFFERS)`.

## Setup

| Parameter | Value |
| --- | --- |
| Dataset | UCI Forest CoverType, ingested as dataset 116 |
| Preprocessing | `covertype` — 10 continuous features standardised, 44 binary indicators kept as 0/1 |
| Dimension | 54 |
| Split | stratified, `seed = 42`, 50,000 train / 3,000 test |
| Training pools | 10,000 and 50,000 rows (stratified subsample of the 50,000-row pool, `seed = 0`) |
| `k` | 5 |
| Distance metric | L2 |
| HNSW build | `m = 16`, `ef_construction = 64` |
| Queries per run | 1,000 (identical query set for both methods at a given size) |
| Repetitions | 3 → 3,000 measured latencies per run |
| Warm-up | 1 discarded query per run |
| Environment | Python 3.12.1, PostgreSQL 18.0, pgvector 0.8.7, 12 CPU cores, single machine, warm buffer cache |

## Raw measurements

| Rows | Method | `ef` | Recall@5 | Accuracy | Macro P | Macro R | Macro F1 | Mean (ms) | p50 (ms) | p95 (ms) | p99 (ms) | Build (ms) | Index (MiB) | Speed-up (p50) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 9,999 | exact | — | 1.0000 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 3.063 | 3.014 | 3.539 | 3.973 | — | — | — |
| 9,999 | hnsw | 10 | 0.9958 | 0.7730 | 0.6302 | 0.4900 | 0.5061 | 0.249 | 0.238 | 0.319 | 0.410 | 620 | 4.98 | 12.6× |
| 9,999 | hnsw | 25 | 0.9996 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 0.273 | 0.262 | 0.356 | 0.426 | 620 | 4.98 | 11.5× |
| 9,999 | hnsw | 50 | 0.9998 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 0.319 | 0.312 | 0.394 | 0.451 | 620 | 4.98 | 9.6× |
| 9,999 | hnsw | 100 | 1.0000 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 0.433 | 0.420 | 0.556 | 0.685 | 620 | 4.98 | 7.2× |
| 9,999 | hnsw | 200 | 1.0000 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 0.637 | 0.620 | 0.809 | 0.936 | 620 | 4.98 | 4.9× |
| 9,999 | hnsw | 400 | 1.0000 | 0.7750 | 0.6310 | 0.4907 | 0.5069 | 0.998 | 0.972 | 1.266 | 1.460 | 620 | 4.98 | 3.1× |
| 50,000 | exact | — | 1.0000 | 0.8560 | 0.7979 | 0.7021 | 0.7314 | 10.920 | 10.903 | 11.384 | 11.891 | — | — | — |
| 50,000 | hnsw | 10 | 0.9822 | 0.8540 | 0.7819 | 0.6887 | 0.7149 | 0.239 | 0.233 | 0.295 | 0.364 | 3,252 | 24.92 | 46.9× |
| 50,000 | hnsw | 25 | 0.9948 | 0.8570 | 0.7982 | 0.7027 | 0.7319 | 0.279 | 0.271 | 0.348 | 0.417 | 3,252 | 24.92 | 40.2× |
| 50,000 | hnsw | 50 | 0.9990 | 0.8560 | 0.7979 | 0.7021 | 0.7314 | 0.343 | 0.334 | 0.438 | 0.528 | 3,252 | 24.92 | 32.6× |
| 50,000 | hnsw | 100 | 1.0000 | 0.8560 | 0.7979 | 0.7021 | 0.7314 | 0.451 | 0.442 | 0.571 | 0.657 | 3,252 | 24.92 | 24.6× |
| 50,000 | hnsw | 200 | 1.0000 | 0.8560 | 0.7979 | 0.7021 | 0.7314 | 0.667 | 0.658 | 0.837 | 0.930 | 3,252 | 24.92 | 16.6× |
| 50,000 | hnsw | 400 | 1.0000 | 0.8560 | 0.7979 | 0.7021 | 0.7314 | 1.121 | 1.105 | 1.407 | 1.539 | 3,252 | 24.92 | 9.9× |

Every HNSW row reports `used_index_scan = True` and the applied
`hnsw.ef_search` equals the requested value. Every exact row reports
`used_sequential_scan = True` — the baseline is exhaustive by construction.

## Findings

### 1. `ef_search` still controls a recall / latency trade-off, but the curve is compressed

Recall@5 rises monotonically with `ef_search` and reaches 1.0000 by
`ef = 100`, while p50 latency rises from 0.238 ms to 0.972 ms at 10,000 rows
and from 0.233 ms to 1.105 ms at 50,000 rows.

The curve is much flatter than on a purely continuous feature space: even
`ef = 10` already retrieves 98.2–99.6 % of the exact neighbours. The reason is
the primary preprocessing configuration — 44 of the 54 coordinates are 0/1
forest-cover indicators, so many Covertype rows sit at identical or nearly
identical positions and the graph neighbourhoods are dense and easy to
traverse. This is a property of *this* dataset, not a general claim about
HNSW; a dataset with 54 continuous dimensions shows a steeper curve.

### 2. Classification is almost completely insensitive to the recall gap

At 10,000 rows, accuracy and macro F1 are **identical** to the exact baseline
for every `ef_search` from 25 upwards (0.7750 / 0.5069). At 50,000 rows the
largest accuracy difference anywhere in the sweep is 0.2 percentage points
(0.8540 at `ef = 10` versus 0.8560 exact), and the largest macro-F1 difference
is 1.65 pp, also at `ef = 10`.

A missed neighbour only changes the prediction if it would have flipped the
majority vote. With `k = 5` the vote is decided by several neighbours, so
classification quality survives a neighbour-set error that Recall@K reports
in full.

### 3. The speed-up grows with the training pool

Exact p50 latency grows from 3.01 ms to 10.90 ms for a 5× larger pool, while
HNSW p50 stays at 0.23–0.33 ms in the low-`ef` region. The measured speed-up
therefore rises from **12.6× to 46.9×** at `ef = 10`, and from **3.1× to 9.9×**
at `ef = 400`.

This is the practical argument for approximate retrieval: it is not a constant
factor, and its advantage widens with table size.

### 4. The build cost and the storage cost are one-off

`CREATE INDEX ... USING hnsw` took 620 ms for 10,000 rows and 3,252 ms for
50,000 rows, producing a 4.98 MiB and a 24.92 MiB index respectively. The
index is reused whenever the metric and build parameters are unchanged, so an
`ef_search` sweep pays that cost exactly once.

### 5. Query latency is a small part of wall-clock time

`api_overhead_s` (total run time minus the sum of measured query latencies)
ranges from 5.2 s to 48.2 s per run. That time is ground-truth computation,
classification, and per-row persistence — not retrieval. Reporting the two
separately is what makes the speed-up figures above meaningful; conflating
them would have hidden the retrieval improvement entirely.

## What changed since the earlier results

The numbers previously published in this folder (a 45× speed-up at
`ef_search = 10`, recall 86.6 %) came from a pipeline that was **not measuring
what it claimed**:

| Defect | Effect |
| --- | --- |
| `SET LOCAL hnsw.ef_search` on an autocommit pool | Silently discarded — every run used pgvector's default of 40, so the "sweep" was flat |
| Nominal/categorical encoding was ordinal | Distances between categories were arbitrary |
| Single timing pass, no repetitions | p95/p99 rested on one sample per query, and the speed-up had no error control |
| No execution-plan check | A sequential-scan fallback would have been reported as HNSW latency |
| `stratified_split(train_fraction=…)` treated the argument as a test fraction | Splits were not the requested size |

All five are fixed, covered by tests, and the sweep above was regenerated from
scratch. The old figures should not be quoted.

## Caveats

* **Tie sensitivity.** Recall@K compares row identifiers. Covertype contains
  duplicate feature vectors, so an equally-valid duplicate returned instead of
  the exact-match row counts as a miss even though the classification is
  identical. This makes Recall@K a *lower* bound on retrieval quality here.
* **Warm buffer cache.** Every run discards one warm-up query; cold-cache
  behaviour is not measured.
* **Single machine.** Absolute latencies describe localhost, not a production
  deployment. Only the *relative* difference is transferable.
* **Per-query round-trip.** Queries are issued one at a time so that a
  per-query latency exists; a batched workload would amortise the round-trip.
* **Class imbalance.** Covertype's classes range from 236 to 24,380 rows in
  the 50,000-row pool, which is why macro F1 accompanies accuracy.

See [`../experiment-plan.md`](../experiment-plan.md) for the full list of
threats to validity.
