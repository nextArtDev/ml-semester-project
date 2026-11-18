/** Shared API types mirroring the FastAPI backend schemas. */

export type DistanceMetric = 'l2' | 'cosine' | 'l1'
export type RetrievalMethod = 'exact' | 'hnsw'
export type PreprocessingMode = 'covertype' | 'standardize_all'

export interface DatasetSummary {
  id: number
  name: string
  source: string
  n_rows: number
  n_features: number
  n_classes: number
  dimension: number
  class_labels: number[]
  preprocessing_mode: PreprocessingMode
  created_at: string
}

export interface IngestResponse {
  dataset_id: number
  name: string
  n_train: number
  n_test: number
  dimension: number
  n_classes: number
  classes: number[]
  class_distribution: number[]
  preprocessing: Record<string, unknown>
  preprocessing_mode: PreprocessingMode
}

export interface PerClassMetric {
  label: number
  precision: number
  recall: number
  f1: number
  support: number
}

export interface Experiment {
  id: number
  dataset_id: number
  name: string | null
  retrieval_method: RetrievalMethod
  k: number
  distance_metric: DistanceMetric
  hnsw_ef_search: number | null
  hnsw_m: number | null
  hnsw_ef_construction: number | null
  train_size: number
  test_size: number
  unique_queries: number
  total_test_pool: number
  index_build_ms: number | null
  index_created_ms: number | null
  index_reused: boolean | null
  index_reused_build_ms: number | null
  index_size_bytes: number | null
  repetitions: number
  recall_at_k: number | null
  tie_aware_recall_at_k: number | null
  tied_queries: number | null
  max_tied_candidates: number | null
  accuracy: number | null
  precision_macro: number | null
  recall_macro: number | null
  macro_f1: number | null
  accuracy_std: number | null
  macro_f1_std: number | null
  recall_at_k_std: number | null
  tie_aware_recall_at_k_std: number | null
  dataset_fingerprint: string | null
  seed: number | null
  latency_avg_ms: number | null
  latency_p50_ms: number | null
  latency_p95_ms: number | null
  latency_p99_ms: number | null
  latency_min_ms: number | null
  latency_max_ms: number | null
  total_runtime_s: number | null
  api_overhead_s: number | null
  total_experiment_runtime_s: number | null
  index_preparation_s: number | null
  index_build_s: number | null
  ground_truth_computation_s: number | null
  retrieval_total_s: number | null
  retrieval_mean_per_query_ms: number | null
  metrics_computation_s: number | null
  persistence_s: number | null
  orchestration_overhead_s: number | null
  query_sampling: 'first' | 'stratified'
  query_sample_seed: number
  evaluated_query_ids: number[]
  test_pool_class_distribution: Record<string, number>
  evaluated_class_distribution: Record<string, number>
  benchmark_round: number | null
  benchmark_position: number | null
  benchmark_seed: number | null
  benchmark_order: string | null
  warm_cache: boolean | null
  used_index_scan: boolean | null
  environment: Record<string, unknown>
  confusion_matrix: number[][]
  per_class_metrics: PerClassMetric[]
  status: string
  error_message: string | null
  created_at: string
}

export interface ExperimentRequest {
  dataset_id: number
  k: number
  distance_metric: DistanceMetric
  method: RetrievalMethod
  ef_search?: number | null
  hnsw_m?: number
  hnsw_ef_construction?: number
  max_queries?: number | null
  repetitions?: number
  name?: string | null
  weighted_voting?: boolean
}

export interface ExperimentComparison {
  exact: Experiment
  approximate: Experiment
  deltas: {
    accuracy: number | null
    precision_macro: number | null
    recall_macro: number | null
    macro_f1: number | null
    recall_at_k: number | null
    latency_avg_ms: number | null
    latency_p50_ms: number | null
    latency_p95_ms: number | null
    speedup: number | null
  }
}

export interface LatencySamples {
  experiment_id: number
  count: number
  samples: number[]
  p50: number
  p90: number
  p95: number
  p99: number
}

export interface HealthStatus {
  status: string
  database: string
  pgvector_version: string | null
  dataset_count: number
  experiment_count: number
}
