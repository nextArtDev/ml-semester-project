# API Reference

Base URL: `http://127.0.0.1:8000` — prefix `/api`.

Interactive documentation is served at `/docs` (Swagger) and `/redoc`.

---

## Health

### `GET /api/health`

```json
{
  "status": "ok",
  "database": "connected",
  "pgvector_version": "0.8.7",
  "dataset_count": 2,
  "experiment_count": 14
}
```

---

## Datasets

### `GET /api/datasets`

Returns every ingested dataset.

### `GET /api/datasets/{id}`

Returns a single dataset. `404` when it does not exist.

### `DELETE /api/datasets/{id}`

Drops the dataset's vector tables (`fv_<id>`, `tv_<id>`) and its row in the
`datasets` table. Cascades to the related experiments.

### `POST /api/datasets/covertype`

Ingests the UCI Forest CoverType dataset.

```json
{
  "name": "Forest CoverType",
  "max_rows": 581012,
  "max_train_rows": null,
  "test_size": 5000,
  "seed": 42
}
```

| Field | Meaning |
| --- | --- |
| `name` | Display name for the dataset |
| `max_rows` | Rows to sample from the source file before splitting (`null` = all) |
| `max_train_rows` | Cap on the training pool, sampled per class (stratified) |
| `test_size` | Number of held-out observations |
| `seed` | Random seed for the split and sampling |

Response:

```json
{
  "dataset_id": 1,
  "name": "Forest CoverType",
  "n_train": 576012,
  "n_test": 5000,
  "dimension": 54,
  "n_classes": 7,
  "classes": [1, 2, 3, 4, 5, 6, 7],
  "class_distribution": [283573, 212801, 35754, 2747, 9493, 17367, 20510],
  "preprocessing": { "...": "fitted state, saved for reproducibility" }
}
```

### `POST /api/datasets/upload`

`multipart/form-data` upload of a CSV file.

| Field | Type | Required |
| --- | --- | --- |
| `file` | file | ✅ |
| `name` | text | ✅ |
| `target_column` | text | — (default `label`) |
| `test_size` | int | — |
| `max_train_rows` | int | — |
| `seed` | int | — |

### `GET /api/datasets/covertype/status`

Reports whether the Covertype archive is cached locally.

---

## Experiments

### `POST /api/experiments/run`

Runs one experiment.

```json
{
  "dataset_id": 1,
  "k": 5,
  "distance_metric": "l2",
  "method": "hnsw",
  "ef_search": 100,
  "hnsw_m": 16,
  "hnsw_ef_construction": 64,
  "max_queries": 2000,
  "repetitions": 3,
  "seed": 0,
  "query_sampling": "stratified",
  "query_sample_seed": 0,
  "tie_tolerance_relative": 1e-6,
  "tie_tolerance_absolute": 1e-6,
  "name": "HNSW ef=100 k=5",
  "weighted_voting": false
}
```

| Field | Type | Notes |
| --- | --- | --- |
| `dataset_id` | int | Required |
| `k` | int | 1–100 |
| `distance_metric` | `l2` \| `cosine` \| `l1` | |
| `method` | `exact` \| `hnsw` | |
| `ef_search` | int \| null | `hnsw.ef_search`, 1–1000. Query-time only. |
| `hnsw_m` | int | Build parameter — changing it forces an index rebuild |
| `hnsw_ef_construction` | int | Build parameter |
| `max_queries` | int \| null | Number of distinct test vectors; `null` means all (the runner maps `--max-queries 0` to `null`) |
| `repetitions` | int | Complete timed passes over that query set, 1–100 |
| `seed` | int | Reproducibility seed recorded on the run |
| `query_sampling` | `first` \| `stratified` | Stratified is the documented default for new artifacts |
| `query_sample_seed` | int | Seed for within-class query sampling |
| `tie_tolerance_relative` | float | Relative term of the boundary tolerance |
| `tie_tolerance_absolute` | float | Absolute term of the boundary tolerance |
| `weighted_voting` | bool | Distance-weighted majority voting |

`repetitions` is not a counter multiplier: the whole query set is executed
`repetitions` times and every latency is stored. A request with
`max_queries = 2000` and `repetitions = 3` stores 6,000 `query_latencies` rows
and 3 `experiment_repetitions` rows. One warm-up query is discarded.
`query_sampling = "stratified"` allocates query slots by class and persists the
selected ids and both class distributions.

Response (abridged):

```json
{
  "id": 15,
  "retrieval_method": "hnsw",
  "k": 5,
  "distance_metric": "l2",
  "hnsw_ef_search": 100,
  "hnsw_m": 16,
  "hnsw_ef_construction": 64,
  "train_size": 15001,
  "test_size": 300,
  "unique_queries": 300,
  "total_test_pool": 3000,
  "repetitions": 3,
  "index_build_ms": null,
  "index_created_ms": null,
  "index_reused": true,
  "index_reused_build_ms": 1973.4,
  "index_size_bytes": 5234688,
  "dataset_fingerprint": "6baa8fa283ea5a8267d3e964eedf5d2a",
  "seed": 0,
  "query_sampling": "stratified",
  "query_sample_seed": 0,
  "evaluated_query_ids": [1, 3, 7, 9, 11],
  "test_pool_class_distribution": {"0": 1094, "1": 1463},
  "evaluated_class_distribution": {"0": 364, "1": 490},
  "benchmark_round": null,
  "benchmark_position": null,
  "benchmark_seed": null,
  "warm_cache": true,
  "recall_at_k": 0.9527,
  "tie_aware_recall_at_k": 0.9527,
  "tied_queries": 0,
  "accuracy": 0.7633,
  "macro_f1": 0.4861,
  "latency_avg_ms": 0.379,
  "latency_p50_ms": 0.3746,
  "latency_p95_ms": 0.4462,
  "used_index_scan": true,
  "total_experiment_runtime_s": 12.34,
  "index_preparation_s": 0.01,
  "index_build_s": 0.0,
  "ground_truth_computation_s": 0.00,
  "ground_truth_cache_hit": true,
  "ground_truth_lookup_s": 0.0001,
  "retrieval_batch_wall_time_s": 1.21,
  "summed_query_execution_time_s": 1.14,
  "retrieval_total_s": 1.14,
  "retrieval_mean_per_query_ms": 0.38,
  "metrics_computation_s": 0.01,
  "persistence_s": 0.05,
  "orchestration_overhead_s": 2.99,
  "confusion_matrix": [[...]],
  "per_class_metrics": [{ "label": 1, "precision": 0.8, "recall": 0.72, "f1": 0.76, "support": 125 }]
}
```

> `index_build_ms` is the duration of a build that actually ran during the
> request. When an index with the same metric and build parameters is reused
> it is `null`, `index_reused` is `true`, and the historical duration is in
> `index_reused_build_ms`. The reported `hnsw_m` / `hnsw_ef_construction` come
> from the live index, not from the request defaults.
>
> The timing fields use `perf_counter`. `summed_query_execution_time_s` is a
> subset of `retrieval_batch_wall_time_s`; only the wall time is part of the
> non-overlapping decomposition. `index_build_s` is 0 when the index was reused,
> and is a subset of `index_preparation_s`.
> The legacy `api_overhead_s` is no longer written: it was total runtime minus
> query time and therefore included ground-truth computation and orchestration.
> `benchmark_*` fields are populated by the batch runner; direct API calls
> leave them `null`.

### `GET /api/experiments/{id}/repetitions`

One row per complete timed pass, including that pass's sample count, latency
distribution, conventional and tie-aware Recall@K, accuracy, macro precision,
macro recall and macro F1. These are the rows behind
`repetition_details.csv`.

### `GET /api/experiments`

Optional query parameters: `dataset_id`, `limit`.

### `GET /api/experiments/{id}`

A single experiment, with the confusion matrix and per-class metrics
deserialised.

### `DELETE /api/experiments/{id}`

### `GET /api/experiments/compare?exact_id={id}&approximate_id={id}`

Returns both experiments plus a `deltas` block:

```json
{
  "exact": { "...": "..." },
  "approximate": { "...": "..." },
  "deltas": {
    "accuracy": -0.0033,
    "macro_f1": 0.0012,
    "recall_at_k": -0.0473,
    "latency_avg_ms": -2.83,
    "latency_p50_ms": -2.84,
    "latency_p95_ms": -3.36,
    "speedup": 8.51
  }
}
```

### `GET /api/experiments/{id}/latencies`

Per-query latency samples with percentile summary.

```json
{
  "experiment_id": 15,
  "count": 900,
  "repetitions": 3,
  "unique_queries": 300,
  "samples": [0.21, 0.19, "..."],
  "p50": 0.3746,
  "p90": 0.42,
  "p95": 0.4462,
  "p99": 0.51
}
```

`count` is the number of stored observations (`unique_queries × repetitions`
when the limit is not reached), and `unique_queries` is derived from the
distinct repetition numbers. The samples are ordered by repetition and query
index, so a client can separate the passes if needed.

### `GET /api/experiments/{id}/explain`

Returns the retrieval configuration of an experiment, including whether the
planner selected an HNSW index scan.
