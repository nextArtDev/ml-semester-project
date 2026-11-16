'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Loader2, Play, Zap } from 'lucide-react'
import { useTranslations } from 'next-intl'
import { useState } from 'react'
import { toast } from 'sonner'

import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Slider } from '@/components/ui/slider'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { listDatasets, runExperiment } from '@/lib/api'
import type { DistanceMetric, RetrievalMethod } from '@/types/api'

export function ExperimentRunner({
  datasetId,
  onDatasetChange,
}: {
  datasetId?: number
  onDatasetChange: (id: number | undefined) => void
}) {
  const t = useTranslations('experiments')
  const tCommon = useTranslations('common')
  const queryClient = useQueryClient()

  const { data: datasets } = useQuery({
    queryKey: ['datasets'],
    queryFn: listDatasets,
  })

  const [method, setMethod] = useState<RetrievalMethod>('exact')
  const [k, setK] = useState(5)
  const [metric, setMetric] = useState<DistanceMetric>('l2')
  const [efSearch, setEfSearch] = useState(100)
  const [hnswM, setHnswM] = useState(16)
  const [efConstruction, setEfConstruction] = useState(64)
  const [maxQueries, setMaxQueries] = useState(1000)
  const [weighted, setWeighted] = useState(false)

  const runMutation = useMutation({
    mutationFn: () => {
      if (!datasetId) throw new Error('Select a dataset first')
      return runExperiment({
        dataset_id: datasetId,
        k,
        distance_metric: metric,
        method,
        ef_search: method === 'hnsw' ? efSearch : null,
        hnsw_m: method === 'hnsw' ? hnswM : 16,
        hnsw_ef_construction: method === 'hnsw' ? efConstruction : 64,
        max_queries: maxQueries,
        weighted_voting: weighted,
        name:
          method === 'hnsw'
            ? `HNSW ef=${efSearch} k=${k}`
            : `Exact k=${k}`,
      })
    },
    onSuccess: (result) => {
      toast.success(
        `${result.name ?? result.id}: ${((result.accuracy ?? 0) * 100).toFixed(2)}% accuracy`,
      )
      queryClient.invalidateQueries({ queryKey: ['experiments'] })
      queryClient.invalidateQueries({ queryKey: ['experiment', result.id] })
    },
    onError: (error: Error) => toast.error(error.message),
  })

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{t('title')}</CardTitle>
        <CardDescription>{t('subtitle')}</CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-5">
        <div className="grid gap-4 sm:grid-cols-3">
          <div className="grid gap-2 sm:col-span-1">
            <Label htmlFor="dataset">{t('dataset')}</Label>
            <Select
              value={datasetId ? String(datasetId) : undefined}
              onValueChange={(value) => onDatasetChange(Number(value))}
            >
              <SelectTrigger id="dataset">
                <SelectValue placeholder={t('selectDataset')} />
              </SelectTrigger>
              <SelectContent>
                {(datasets ?? []).map((ds) => (
                  <SelectItem key={ds.id} value={String(ds.id)}>
                    {ds.name} ({ds.dimension}d)
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="grid gap-2">
            <Label htmlFor="k">{t('k')}</Label>
            <Input
              id="k"
              type="number"
              min={1}
              max={100}
              value={k}
              onChange={(e) => setK(Number(e.target.value))}
            />
          </div>

          <div className="grid gap-2">
            <Label htmlFor="metric">{t('metric')}</Label>
            <Select
              value={metric}
              onValueChange={(value) => setMetric(value as DistanceMetric)}
            >
              <SelectTrigger id="metric">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="l2">L2 (Euclidean)</SelectItem>
                <SelectItem value="cosine">Cosine</SelectItem>
                <SelectItem value="l1">L1 (Manhattan)</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>

        <Separator />

        <Tabs
          value={method}
          onValueChange={(value) => setMethod(value as RetrievalMethod)}
        >
          <TabsList className="w-full sm:w-auto">
            <TabsTrigger value="exact" className="gap-2">
              <Play className="size-4" />
              {tCommon('exact')}
            </TabsTrigger>
            <TabsTrigger value="hnsw" className="gap-2">
              <Zap className="size-4" />
              {tCommon('approximate')}
            </TabsTrigger>
          </TabsList>
        </Tabs>

        <p className="-mt-2 text-sm text-muted-foreground">
          {method === 'exact' ? t('methodExactHint') : t('methodHnswHint')}
        </p>

        {method === 'hnsw' && (
          <div className="grid gap-5 rounded-lg border border-border/70 bg-muted/25 p-4 sm:grid-cols-3">
            <div className="grid gap-3">
              <div className="flex items-baseline justify-between gap-2">
                <Label htmlFor="ef">{t('efSearch')}</Label>
                <span className="text-sm font-semibold tabular-nums text-primary">
                  {efSearch}
                </span>
              </div>
              <Slider
                id="ef"
                min={5}
                max={800}
                step={5}
                value={[efSearch]}
                onValueChange={([value]) => setEfSearch(value)}
              />
              <p className="text-xs text-muted-foreground">{t('efSearchHint')}</p>
            </div>

            <div className="grid gap-2">
              <Label htmlFor="hnsw-m">{t('hnswM')}</Label>
              <Input
                id="hnsw-m"
                type="number"
                min={2}
                max={128}
                value={hnswM}
                onChange={(e) => setHnswM(Number(e.target.value))}
              />
            </div>

            <div className="grid gap-2">
              <Label htmlFor="efc">{t('hnswEfConstruction')}</Label>
              <Input
                id="efc"
                type="number"
                min={4}
                max={1000}
                value={efConstruction}
                onChange={(e) => setEfConstruction(Number(e.target.value))}
              />
            </div>
          </div>
        )}

        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="grid gap-2">
            <Label htmlFor="queries">{t('maxQueries')}</Label>
            <Input
              id="queries"
              type="number"
              min={1}
              max={50000}
              value={maxQueries}
              onChange={(e) => setMaxQueries(Number(e.target.value))}
              className="w-40"
            />
          </div>

          <div className="flex items-center gap-3">
            <label className="flex cursor-pointer items-center gap-2 text-sm text-muted-foreground">
              <input
                type="checkbox"
                checked={weighted}
                onChange={(e) => setWeighted(e.target.checked)}
                className="size-4 rounded border-border accent-primary"
              />
              {t('weightedVoting')}
            </label>

            <Button
              onClick={() => runMutation.mutate()}
              disabled={!datasetId || runMutation.isPending}
            >
              {runMutation.isPending && (
                <Loader2 className="size-4 animate-spin" />
              )}
              {runMutation.isPending
                ? t('running')
                : method === 'hnsw'
                  ? t('runHnsw')
                  : t('runExact')}
            </Button>
          </div>
        </div>

        {!datasets || datasets.length === 0 ? (
          <p className="text-sm text-muted-foreground">{t('noDatasets')}</p>
        ) : null}
      </CardContent>
    </Card>
  )
}
