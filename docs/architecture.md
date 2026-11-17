# Architecture Notes

## Why the exact baseline uses a *function* and not an *operator*

pgvector exposes two equivalent ways to express a distance:

| Form | Expression | Index-eligible |
| --- | --- | --- |
| Operator | `embedding <-> $1` | ✅ yes — this is what HNSW satisfies |
| Function | `l2_distance(embedding, $1)` | ❌ no — always a sequential scan |

If the exact baseline used the operator form, the planner could opportunistically
pick the HNSW index whenever one existed, and the "baseline" would silently
become an approximate search. Using the function form guarantees an exhaustive
scan *by construction*, independent of the index state.

Verified with `EXPLAIN (ANALYZE)`:

```
Limit  (cost=948.72..948.73 rows=5 width=18) (actual time=8.850..8.851 rows=5.00 loops=1)
  ->  Sort  (cost=948.67..986.18 rows=15001 width=18) (actual time=8.847..8.848 rows=5.00 loops=1)
        Sort Key: (l2_distance(fv_1.embedding, ...))
        Sort Method: top-N heapsort
        ->  Seq Scan on fv_1  (cost=0.00..699.51 rows=15001 width=18) (actual time=1.448..6.938 rows=15001.00 loops=1)
```

## Why one vector table per dataset

Two independent pgvector constraints force this design:

1. **A fixed dimension is mandatory.** `CREATE INDEX ... USING hnsw (embedding)`
   fails with *"column does not have dimensions"* unless the column is declared
   as `vector(dim)`. A single shared table would need one column per possible
   dimension.

2. **Residual filters are applied after the index scan.** pgvector's HNSW scan
   retrieves candidate neighbours from the whole table and only *then* applies
   the `WHERE dataset_id = ...` filter. On a multi-dataset table this returns
   neighbours from the wrong dataset, which silently corrupts Recall@K.

The platform therefore creates `fv_<dataset_id>` and `tv_<dataset_id>` per
dataset. Both use a dimensionally-typed `vector(dim)` column, so the planner
produces a clean HNSW index scan with no residual filter.

## Autocommit and `SET LOCAL`

The connection pool runs with `autocommit=True`. In that mode PostgreSQL emits

```
WARNING:  SET LOCAL can only be used in transaction blocks
```

and the setting is **silently discarded**. The first implementation of the
`ef_search` sweep used `SET LOCAL hnsw.ef_search = ...`, and every run used the
default value of 40 — which is why the initial sweep produced a flat, noisy
recall curve.

The fix is to use the session-level `SET` and explicitly `RESET` it in a
`finally` block, so a pooled connection never leaks the setting into the next
request.

## Warm-up query

One warm-up query is executed immediately before the first timed query of a
batch and then discarded. This removes first-call planning cost and
cold-buffer effects from the first measurement. Later repetitions of the same
query set do **not** repeat the warm-up: the cache and the plan are already
warm, and discarding a measured query from every pass would silently shrink
the stored sample count.

## Repetitions

`repetitions` is the number of complete timed passes over the query set, not a
multiplier applied to a counter. Every pass executes every query, every
latency is stored with its repetition number, and the persisted row count is
exactly `repetitions × unique_queries`. The query order is identical in every
pass, so the repetitions measure run-to-run variability (timing, buffer state,
graph traversal) rather than population sampling. Per-pass metrics are stored
in `experiment_repetitions`; they are not inferred from the aggregate row.

## Index reuse

Building an HNSW index over 50,000 rows took 3.24 s on the evaluation machine.
Since `ef_search` is a *query-time* parameter, a sweep over many `ef_search`
values must not rebuild the index each time.

The `hnsw_indexes` registry table records the metric and build parameters of
the index that exists for a dataset. `ensure_hnsw_index` returns an
`IndexInfo` describing what actually happened: whether a build ran, its
measured duration, the live index's `m`/`ef_construction`, and its size. It
reuses an index only when the metric and both build parameters match *and* the
physical index is present, and rebuilds otherwise.

A reused index has **no** current build time. The experiment stores
`index_build_ms = NULL`, `index_reused = True`, and the historical duration in
`index_reused_build_ms`. Reporting the historical figure as a cost paid by
each `ef_search` configuration would overstate the sweep's cost.

The exact baseline deliberately **does not drop** the index. Because the
function form cannot use it anyway, leaving it in place means a following HNSW
run reuses it instead of paying the rebuild cost.

## Data flow

```
CSV / Covertype
      │
      ▼
stratified_split  ──────────────► train_df            test_df
      │                              │                   │
      │                    fit_preprocessor(train_df)    │  ← statistics from
      │                              │                   │    training only
      │                              ▼                   │
      │              standardise 10 continuous,          │
      │              keep 44 indicators 0/1              │
      │              (one-hot nominal for uploads)       │
      │                              │                   │
      │                              ▼                   ▼
      │                    transform(train)      transform(test)
      │                              │                   │
      ▼                              ▼                   ▼
  datasets row            COPY → fv_<id>       COPY → tv_<id>
                                   │
                    ┌──────────────┴───────────────┐
                    ▼                              ▼
           exact (function form)          HNSW (operator form)
           ground truth + baseline       recall, latency, accuracy
                    │                              │
                    └──────────────┬───────────────┘
                                   ▼
                        majority vote → metrics → experiments
```

## Ground truth and ties

The reference neighbours always come from the exhaustive query using the
distance *function*, whatever method is under test. Two properties make the
reference reproducible:

* the exact top-k is ordered by `(distance, id)`, so the K-th row is chosen
  deterministically when distances are equal;
* a second query returns every id whose distance is within a documented
  floating-point tolerance of the K-th boundary distance
  (`boundary × (1 + 1e-6) + 1e-6`).

Conventional Recall@K compares retrieved ids against the deterministic top-k.
Tie-aware Recall@K counts any retrieved id inside the boundary candidate set
as valid and divides by `k`. The tolerance and both metrics are persisted per
experiment. On a pool with no boundary ties the two metrics coincide, which is
a measurable outcome rather than an assumption.

## Measurement order

Within a single experiment run:

1. The HNSW index is created (or reused) and `ANALYZE` refreshes planner stats.
2. The **exact** ground-truth ids and boundary candidate sets are computed for
   every query.
3. The method under test runs `repetitions` complete passes over the query
   set, with one warm-up query before the first pass.
4. Conventional and tie-aware Recall@K are computed **per repetition** from
   that pass's neighbour sets, as are accuracy, macro precision/recall/F1 and
   the confusion matrix.
5. Both neighbour sets are fed through the *same* majority-voting function, so
   any accuracy difference is attributable purely to the retrieval stage.
6. The aggregate row is written, then one `experiment_repetitions` row per
   pass and one `query_latencies` row per measured query.
