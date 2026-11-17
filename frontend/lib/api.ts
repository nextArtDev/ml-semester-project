import type {
  DatasetSummary,
  Experiment,
  ExperimentComparison,
  ExperimentRequest,
  HealthStatus,
  IngestResponse,
  LatencySamples,
} from '@/types/api'

export const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') ?? 'http://127.0.0.1:8000'

const API_PREFIX = '/api'

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${API_PREFIX}${path}`, {
    ...init,
    headers: {
      'Content-Type': 'application/json',
      ...init?.headers,
    },
    cache: 'no-store',
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const payload = await response.json()
      detail = payload.detail ?? JSON.stringify(payload)
    } catch {
      /* response body was not JSON */
    }
    throw new ApiError(detail, response.status)
  }

  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

// --- Health ----------------------------------------------------------------
export const getHealth = () => request<HealthStatus>('/health')

// --- Datasets --------------------------------------------------------------
export const listDatasets = () => request<DatasetSummary[]>('/datasets')

export const getDataset = (id: number) => request<DatasetSummary>(`/datasets/${id}`)

export const deleteDataset = (id: number) =>
  request<{ deleted: number }>(`/datasets/${id}`, { method: 'DELETE' })

export const ingestCovertype = (body: {
  name?: string
  max_rows?: number | null
  test_size?: number | null
  max_train_rows?: number | null
  seed?: number
  preprocessing_mode?: 'covertype' | 'standardize_all'
}) =>
  request<IngestResponse>('/datasets/covertype', {
    method: 'POST',
    body: JSON.stringify(body),
  })

export const uploadDataset = async (form: FormData) => {
  const response = await fetch(`${API_BASE_URL}${API_PREFIX}/datasets/upload`, {
    method: 'POST',
    body: form,
  })

  if (!response.ok) {
    let detail = response.statusText
    try {
      const payload = await response.json()
      detail = payload.detail ?? JSON.stringify(payload)
    } catch {
      /* response body was not JSON */
    }
    throw new ApiError(detail, response.status)
  }
  return (await response.json()) as IngestResponse
}

// --- Experiments ------------------------------------------------------------
export const listExperiments = (datasetId?: number) =>
  request<Experiment[]>(
    `/experiments${datasetId ? `?dataset_id=${datasetId}` : ''}`,
  )

export const getExperiment = (id: number) => request<Experiment>(`/experiments/${id}`)

export const deleteExperiment = (id: number) =>
  request<{ deleted: number }>(`/experiments/${id}`, { method: 'DELETE' })

export const runExperiment = (body: ExperimentRequest) =>
  request<Experiment>('/experiments/run', {
    method: 'POST',
    body: JSON.stringify(body),
  })

export const compareExperiments = (exactId: number, approximateId: number) =>
  request<ExperimentComparison>(
    `/experiments/compare?exact_id=${exactId}&approximate_id=${approximateId}`,
  )

export const getLatencySamples = (id: number) =>
  request<LatencySamples>(`/experiments/${id}/latencies`)
