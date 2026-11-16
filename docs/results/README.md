# Experimental Results

Raw artefacts produced by `backend/scripts/run_experiments.py`:

* [`sweep.csv`](sweep.csv) — one row per experiment
* [`sweep.json`](sweep.json) — the same data as JSON

## Setup

| Parameter | Value |
| --- | --- |
| Dataset | UCI Forest CoverType (54 features, 7 classes) |
| Training pool | 10,000 and 50,000 rows (stratified subsample of the full dataset) |
| Test pool | 3,000 rows |
| `k` | 5 |
| Distance metric | L2 |
| HNSW build | `m = 16`, `ef_construction = 64` |
| Queries per run | 1,000 |
| Hardware | Single machine, PostgreSQL 18 + pgvector 0.8.7, local socket |

## Results

```
    size   method   k    ef   recall      acc  macroF1     p50ms     p95ms   buildms
------------------------------------------------------------------------------------
   10000    exact   5     —   1.0000   0.7500   0.6256     2.490     2.852         —
   50000    exact   5     —   1.0000   0.8430   0.7580    11.306    11.783         —
   10000     hnsw   5    10   0.8738   0.7190   0.4940     0.220     0.272     560.3
   10000     hnsw   5    25   0.9162   0.7280   0.5264     0.242     0.299     560.3
   10000     hnsw   5    50   0.9468   0.7370   0.5606     0.281     0.359     560.3
   10000     hnsw   5   100   0.9630   0.7420   0.5708     0.350     0.461     560.3
   10000     hnsw   5   200   0.9756   0.7440   0.5860     0.481     0.615     560.3
   10000     hnsw   5   400   0.9850   0.7460   0.6221     0.783     0.984     560.3
   50000     hnsw   5    10   0.8658   0.8100   0.6706     0.251     0.310   3080.0
   50000     hnsw   5    25   0.9046   0.8230   0.7169     0.275     0.356   3080.0
   50000     hnsw   5    50   0.9294   0.8310   0.7425     0.354     0.457   3080.0
   50000     hnsw   5   100   0.9460   0.8300   0.7393     0.479     0.618   3080.0
   50000     hnsw   5   200   0.9562   0.8360   0.7598     0.620     0.822   3080.0
   50000     hnsw   5   400   0.9654   0.8400   0.7662     1.002     1.294   3080.0
```

### Speed-up over the exact baseline

| Training rows | `ef_search` | Recall@K | Accuracy | Δ accuracy | p50 latency | **Speed-up** |
| --- | --- | --- | --- | --- | --- | --- |
| 10,000 | 10 | 87.4% | 71.9% | −3.10 pp | 0.220 ms | **11.3×** |
| 10,000 | 100 | 96.3% | 74.2% | −0.80 pp | 0.350 ms | **7.1×** |
| 10,000 | 400 | 98.5% | 74.6% | −0.40 pp | 0.783 ms | **3.2×** |
| 50,000 | 10 | 86.6% | 81.0% | −3.30 pp | 0.251 ms | **45.0×** |
| 50,000 | 100 | 94.6% | 83.0% | −1.30 pp | 0.479 ms | **23.6×** |
| 50,000 | 400 | 96.5% | 84.0% | −0.30 pp | 1.002 ms | **11.3×** |

## Findings

### 1. The search-effort parameter controls a clean recall / latency trade-off

Recall@K rises monotonically with `ef_search` at every dataset size, while
latency rises alongside it. The curve has the classic shape of an
approximate-nearest-neighbour search: a steep recall gain for the first
order of magnitude of `ef_search`, then diminishing returns.

### 2. Classification quality is far more robust than retrieval recall

The gap between exact and approximate **accuracy** is much smaller than the
gap between exact and approximate **recall**:

* at `ef_search = 10`, Recall@K is 86.6 % but accuracy is only 3.3 pp lower
* at `ef_search = 400`, Recall@K is 96.5 % and accuracy is only 0.3 pp lower

This is the key practical result. A missed neighbour only changes the
predicted class if it would have flipped the majority vote. Because the
`k = 5` vote is decided by several neighbours, the classification outcome is
far more stable than the neighbour set itself.

### 3. The benefit of approximate retrieval grows with the dataset

Exact-search latency grows roughly linearly with the table size
(2.49 ms → 11.31 ms for a 5× increase in rows), while HNSW latency grows
much more slowly (0.220 ms → 0.251 ms). The speed-up therefore widens from
**11.3×** at 10,000 rows to **45.0×** at 50,000 rows.

The practical conclusion is that approximate retrieval is not merely a
constant-factor optimisation: the larger the dataset, the more compelling it
becomes.

### 4. The index build cost is a one-off investment

Building the HNSW index took 560 ms for 10,000 rows and 3,080 ms for 50,000
rows. Amortised over a long enough sequence of queries, the build cost is
recovered quickly, and the platform **reuses the index** whenever the metric
and build parameters are unchanged — so an `ef_search` sweep does not pay
the construction cost again.

### 5. A single index serves many configurations

`ef_search` is a *query-time* parameter. Changing it does not require
rebuilding the index, so a practitioner can tune the trade-off per request
without any DDL.

## Caveats

See [`../experiment-plan.md`](../experiment-plan.md) for the full list of
threats to validity. The most important ones here are the class imbalance of
CoverType (which is why macro F1 is reported alongside accuracy) and the warm
buffer cache (every measurement follows a discarded warm-up query).
