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
| `max_queries` | int \| null | Number of test vectors to query |
| `weighted_voting` | bool | Distance-weighted majority voting |

Response (abridged):

```json
{
  "id": 15,
  "retrieval_method": "hnsw",
  "k": 5,
  "distance_metric": "l2",
  "hnsw_ef_search": 100,
  "train_size": 15001,
  "test_size": 300,
  "index_build_ms": 1973.4,
  "recall_at_k": 0.9527,
  "accuracy": 0.7633,
  "macro_f1": 0.4861,
  "latency_avg_ms": 0.379,
  "latency_p50_ms": 0.3746,
  "latency_p95_ms": 0.4462,
  "used_index_scan": true,
  "confusion_matrix": [[...]],
  "per_class_metrics": [{ "label": 1, "precision": 0.8, "recall": 0.72, "f1": 0.76, "support": 125 }]
}
```

> The `index_build_ms` value is the duration of the build that actually ran.
> When an existing index with the same parameters is reused, the original
> build time is reported instead.

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
  "count": 300,
  "samples": [0.21, 0.19, "..."],
  "p50": 0.3746,
  "p90": 0.42,
  "p95": 0.4462,
  "p99": 0.51
}
```

### `GET /api/experiments/{id}/explain`

Returns the retrieval configuration of an experiment, including whether the
planner selected an HNSW index scan.
