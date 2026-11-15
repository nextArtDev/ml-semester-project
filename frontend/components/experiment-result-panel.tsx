'use client'

import { useQuery } from '@tanstack/react-query'
import { AlertCircle, CheckCircle2 } from 'lucide-react'
import { useTranslations } from 'next-intl'

import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { getExperiment, getLatencySamples } from '@/lib/api'
import { cn, formatMs, formatPercent } from '@/lib/utils'

export function ExperimentResultPanel({
  experimentId,
}: {
  experimentId: number | undefined
}) {
  const t = useTranslations('experiments')
  const tCommon = useTranslations('common')

  const { data: experiment, isLoading } = useQuery({
    queryKey: ['experiment', experimentId],
    queryFn: () => getExperiment(experimentId!),
    enabled: experimentId !== undefined,
  })

  const { data: latencies } = useQuery({
    queryKey: ['latencies', experimentId],
    queryFn: () => getLatencySamples(experimentId!),
    enabled: experimentId !== undefined,
  })

  if (!experimentId) {
    return (
      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t('results')}</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            {tCommon('loading')} — {t('viewResults')}
          </p>
        </CardContent>
      </Card>
    )
  }

  if (isLoading || !experiment) {
    return (
      <Card>
        <CardHeader>
          <Skeleton className="h-5 w-32" />
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-16" />
          ))}
        </CardContent>
      </Card>
    )
  }

  const stats = [
    { label: t('accuracy'), value: formatPercent(experiment.accuracy) },
    { label: t('macroF1'), value: formatPercent(experiment.macro_f1) },
    { label: t('recallAtK'), value: formatPercent(experiment.recall_at_k) },
    { label: t('latencyP50'), value: formatMs(experiment.latency_p50_ms) },
    { label: t('latencyP95'), value: formatMs(experiment.latency_p95_ms) },
    {
      label: t('indexBuild'),
      value: experiment.index_build_ms
        ? formatMs(experiment.index_build_ms)
        : tCommon('none'),
    },
  ]

  return (
    <div className="flex flex-col gap-6">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-center justify-between gap-3">
            <CardTitle className="text-base">
              {experiment.name ?? `#${experiment.id}`}
            </CardTitle>
            <div className="flex items-center gap-2">
              <Badge
                variant={
                  experiment.retrieval_method === 'hnsw' ? 'default' : 'secondary'
                }
              >
                {experiment.retrieval_method === 'hnsw'
                  ? tCommon('approximate')
                  : tCommon('exact')}
              </Badge>
              {experiment.retrieval_method === 'hnsw' &&
                (experiment.used_index_scan ? (
                  <span className="inline-flex items-center gap-1 text-xs text-emerald-600 dark:text-emerald-400">
                    <CheckCircle2 className="size-3.5" />
                    {t('indexUsed')}
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400">
                    <AlertCircle className="size-3.5" />
                    {t('indexNotUsed')}
                  </span>
                ))}
            </div>
          </div>
          <CardDescription>
            k={experiment.k} · {experiment.distance_metric.toUpperCase()}
            {experiment.hnsw_ef_search
              ? ` · ef_search=${experiment.hnsw_ef_search}`
              : ''}{' '}
            · {t('trainSize')}: {experiment.train_size.toLocaleString()} ·{' '}
            {t('testSize')}: {experiment.test_size.toLocaleString()}
          </CardDescription>
        </CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-3 lg:grid-cols-6">
          {stats.map((stat) => (
            <div
              key={stat.label}
              className="rounded-lg border border-border/70 bg-muted/25 p-3"
            >
              <div className="text-[11px] uppercase tracking-wide text-muted-foreground">
                {stat.label}
              </div>
              <div className="mt-1 text-lg font-semibold tabular-nums">
                {stat.value}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('confusionMatrix')}</CardTitle>
            <CardDescription>
              {t('trueLabel')} (rows) × {t('predLabel')} (columns)
            </CardDescription>
          </CardHeader>
          <CardContent className="overflow-x-auto">
            {experiment.confusion_matrix.length === 0 ? (
              <p className="text-sm text-muted-foreground">—</p>
            ) : (
              <Table className="w-fit">
                <TableHeader>
                  <TableRow>
                    <TableHead className="w-16" />
                    {experiment.confusion_matrix[0].map((_, i) => (
                      <TableHead key={i} className="text-center tabular-nums">
                        {i + 1}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {experiment.confusion_matrix.map((row, i) => (
                    <TableRow key={i}>
                      <TableCell className="font-medium tabular-nums">
                        {i + 1}
                      </TableCell>
                      {row.map((value, j) => {
                        const total = experiment.confusion_matrix.reduce(
                          (sum, r) => sum + r[j],
                          0,
                        )
                        const intensity = total > 0 ? value / total : 0
                        return (
                          <TableCell
                            key={j}
                            className={cn(
                              'text-center tabular-nums',
                              value > 0 && intensity > 0.5 && 'font-semibold',
                            )}
                            style={
                              intensity > 0
                                ? {
                                    backgroundColor: `color-mix(in oklab, var(--chart-1) ${Math.round(intensity * 45)}%, transparent)`,
                                  }
                                : undefined
                            }
                          >
                            {value}
                          </TableCell>
                        )
                      })}
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('perClass')}</CardTitle>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>{t('trueLabel')}</TableHead>
                  <TableHead className="text-end">P</TableHead>
                  <TableHead className="text-end">R</TableHead>
                  <TableHead className="text-end">F1</TableHead>
                  <TableHead className="text-end">{t('support')}</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {experiment.per_class_metrics.map((row) => (
                  <TableRow key={row.label}>
                    <TableCell className="font-medium tabular-nums">
                      {row.label}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {formatPercent(row.precision, 1)}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {formatPercent(row.recall, 1)}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {formatPercent(row.f1, 1)}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {row.support}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      </div>

      {latencies && latencies.samples.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base">
              {t('latencyDistribution')}
            </CardTitle>
            <CardDescription>
              p50 {formatMs(latencies.p50)} · p90 {formatMs(latencies.p90)} · p95{' '}
              {formatMs(latencies.p95)} · p99 {formatMs(latencies.p99)} · n=
              {latencies.count}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <LatencyHistogram samples={latencies.samples} />
          </CardContent>
        </Card>
      )}
    </div>
  )
}

function LatencyHistogram({ samples }: { samples: number[] }) {
  const bins = 24
  const min = Math.min(...samples)
  const max = Math.max(...samples)
  const width = (max - min) / bins || 1

  const counts = new Array(bins).fill(0) as number[]
  for (const value of samples) {
    const idx = Math.min(bins - 1, Math.floor((value - min) / width))
    counts[idx] += 1
  }
  const peak = Math.max(...counts)

  return (
    <div className="flex h-40 items-end gap-0.5">
      {counts.map((count, i) => (
        <div
          key={i}
          className="flex-1 rounded-t bg-chart-1/70 transition-colors hover:bg-chart-1"
          style={{ height: `${Math.max(2, (count / peak) * 100)}%` }}
          title={`${count} queries`}
        />
      ))}
    </div>
  )
}
