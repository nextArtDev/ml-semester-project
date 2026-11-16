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

The first query of every run is executed twice and only the second execution
is timed. This removes first-call planning cost and cold-buffer effects from
the measurement. Without it the very first query of a run reports a latency
an order of magnitude above the rest.

## Index reuse

Building an HNSW index over 15 000 rows takes roughly 2 seconds. Since
`ef_search` is a *query-time* parameter, a sweep over many `ef_search` values
must not rebuild the index each time.

The `hnsw_indexes` registry table records the metric and build parameters of
the index that exists for a dataset. `create_hnsw_index` reuses it when the
parameters match, and rebuilds only when they change.

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
      │                    StandardScaler + ordinal      │
      │                    encoding                     │
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

## Measurement order

Within a single experiment run:

1. The HNSW index is created (or reused) and `ANALYZE` refreshes planner stats.
2. The **exact** ground-truth neighbour set is computed for every query.
3. The method under test is run for every query, with a warm-up query first.
4. Recall@K compares the retrieved ids against the ground-truth ids.
5. Both neighbour sets are fed through the *same* majority-voting function, so
   any accuracy difference is attributable purely to the retrieval stage.
