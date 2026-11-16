'use client'

import { useQuery } from '@tanstack/react-query'
import { useTranslations } from 'next-intl'
import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { listExperiments } from '@/lib/api'
import type { Experiment } from '@/types/api'
import { cn, formatPercent } from '@/lib/utils'

const CHART_COLORS = ['var(--chart-1)', 'var(--chart-2)']

export function CompareContent() {
  const t = useTranslations('compare')
  const tExp = useTranslations('experiments')

  const { data: experiments } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => listExperiments(),
  })

  const [exactId, setExactId] = useState<string>('')
  const [approxId, setApproxId] = useState<string>('')

  const exactOptions = useMemo(
    () => (experiments ?? []).filter((e) => e.retrieval_method === 'exact'),
    [experiments],
  )
  const approxOptions = useMemo(
    () => (experiments ?? []).filter((e) => e.retrieval_method === 'hnsw'),
    [experiments],
  )

  const exact = exactOptions.find((e) => String(e.id) === exactId)
  const approx = approxOptions.find((e) => String(e.id) === approxId)

  const sweep = useMemo(() => {
    const rows = (experiments ?? [])
      .filter((e) => e.retrieval_method === 'hnsw' && e.hnsw_ef_search)
      .sort((a, b) => (a.hnsw_ef_search ?? 0) - (b.hnsw_ef_search ?? 0))
    return rows.map((e) => ({
      ef: e.hnsw_ef_search ?? 0,
      recall: Number(((e.recall_at_k ?? 0) * 100).toFixed(2)),
      accuracy: Number(((e.accuracy ?? 0) * 100).toFixed(2)),
      p50: Number((e.latency_p50_ms ?? 0).toFixed(3)),
    }))
  }, [experiments])

  const comparisonRows = exact && approx ? buildComparison(exact, approx) : []

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <h1 className="text-3xl font-semibold tracking-tight">{t('title')}</h1>
        <p className="max-w-3xl text-muted-foreground">{t('subtitle')}</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {t('exactExperiment')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Select value={exactId} onValueChange={setExactId}>
              <SelectTrigger>
                <SelectValue placeholder={tExp('selectTwo')} />
              </SelectTrigger>
              <SelectContent>
                {exactOptions.map((e) => (
                  <SelectItem key={e.id} value={String(e.id)}>
                    {e.name ?? `#${e.id}`} â€” {formatPercent(e.accuracy)}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium text-muted-foreground">
              {t('approxExperiment')}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <Select value={approxId} onValueChange={setApproxId}>
              <SelectTrigger>
                <SelectValue placeholder={tExp('selectTwo')} />
              </SelectTrigger>
              <SelectContent>
                {approxOptions.map((e) => (
                  <SelectItem key={e.id} value={String(e.id)}>
                    {e.name ?? `#${e.id}`} â€” ef={e.hnsw_ef_search}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </CardContent>
        </Card>
      </div>

      {!exact || !approx ? (
        <Card>
          <CardContent className="py-10 text-center text-sm text-muted-foreground">
            {t('noSelection')}
          </CardContent>
        </Card>
      ) : (
        <>
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t('metric')}</CardTitle>
              <CardDescription>
                {exact.name ?? `#${exact.id}`} vs {approx.name ?? `#${approx.id}`}
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>{t('metric')}</TableHead>
                    <TableHead className="text-end">{t('exactValue')}</TableHead>
                    <TableHead className="text-end">{t('approxValue')}</TableHead>
                    <TableHead className="text-end">{t('delta')}</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {comparisonRows.map((row) => (
                    <TableRow key={row.label}>
                      <TableCell>{row.label}</TableCell>
                      <TableCell className="text-end tabular-nums">
                        {row.exact}
                      </TableCell>
                      <TableCell className="text-end tabular-nums">
                        {row.approx}
                      </TableCell>
                      <TableCell
                        className={cn(
                          'text-end tabular-nums font-medium',
                          row.positive === true && 'text-emerald-600 dark:text-emerald-400',
                          row.positive === false && 'text-destructive',
                        )}
                      >
                        {row.delta}
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t('conclusion')}</CardTitle>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-3">
              <ConclusionTile
                label={t('recallImpact')}
                value={formatPercent(approx.recall_at_k)}
                tone={
                  (approx.recall_at_k ?? 0) >= 0.95
                    ? 'good'
                    : (approx.recall_at_k ?? 0) >= 0.85
                      ? 'warn'
                      : 'bad'
                }
              />
              <ConclusionTile
                label={t('accuracyImpact')}
                value={`${(approx.accuracy ?? 0) * 100 >= (exact.accuracy ?? 0) * 100 ? '+' : ''}${(((approx.accuracy ?? 0) - (exact.accuracy ?? 0)) * 100).toFixed(2)} pp`}
                tone={
                  Math.abs((approx.accuracy ?? 0) - (exact.accuracy ?? 0)) < 0.01
                    ? 'good'
                    : (approx.accuracy ?? 0) > (exact.accuracy ?? 0)
                      ? 'good'
                      : 'warn'
                }
              />
              <ConclusionTile
                label={t('speedup')}
                value={
                  exact.latency_avg_ms && approx.latency_avg_ms
                    ? `${(exact.latency_avg_ms / approx.latency_avg_ms).toFixed(2)}Ã—`
                    : 'â€”'
                }
                tone="good"
              />
            </CardContent>
          </Card>
        </>
      )}

      {sweep.length > 0 && (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Recall / latency vs ef_search
              </CardTitle>
              <CardDescription>
                The classic search-effort trade-off curve.
              </CardDescription>
            </CardHeader>
            <CardContent className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={sweep}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                  <XAxis
                    dataKey="ef"
                    tick={{ fontSize: 11 }}
                    label={{
                      value: 'ef_search',
                      position: 'insideBottom',
                      offset: -5,
                      fontSize: 11,
                    }}
                  />
                  <YAxis
                    yAxisId="recall"
                    tick={{ fontSize: 11 }}
                    domain={[0, 100]}
                  />
                  <YAxis
                    yAxisId="latency"
                    orientation="right"
                    tick={{ fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{
                      fontSize: 12,
                      background: 'var(--popover)',
                      border: '1px solid var(--border)',
                      borderRadius: 8,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line
                    yAxisId="recall"
                    type="monotone"
                    dataKey="recall"
                    name="Recall@K (%)"
                    stroke={CHART_COLORS[0]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                  <Line
                    yAxisId="latency"
                    type="monotone"
                    dataKey="p50"
                    name="Latency p50 (ms)"
                    stroke={CHART_COLORS[1]}
                    strokeWidth={2}
                    dot={{ r: 3 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-base">
                Accuracy vs ef_search
              </CardTitle>
              <CardDescription>
                Classification quality under varying search effort.
              </CardDescription>
            </CardHeader>
            <CardContent className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={sweep}>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
                  <XAxis dataKey="ef" tick={{ fontSize: 11 }} />
                  <YAxis
                    tick={{ fontSize: 11 }}
                    domain={[0, 100]}
                    label={{
                      value: 'Accuracy (%)',
                      angle: -90,
                      position: 'insideLeft',
                      fontSize: 11,
                    }}
                  />
                  <Tooltip
                    contentStyle={{
                      fontSize: 12,
                      background: 'var(--popover)',
                      border: '1px solid var(--border)',
                      borderRadius: 8,
                    }}
                  />
                  <Bar dataKey="accuracy" radius={[6, 6, 0, 0]}>
                    {sweep.map((_, i) => (
                      <Cell key={i} fill={CHART_COLORS[0]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </div>
      )}
    </div>
  )
}

function ConclusionTile({
  label,
  value,
  tone,
}: {
  label: string
  value: string
  tone: 'good' | 'warn' | 'bad'
}) {
  return (
    <div
      className={cn(
        'rounded-lg border p-4',
        tone === 'good' && 'border-emerald-500/30 bg-emerald-500/5',
        tone === 'warn' && 'border-amber-500/30 bg-amber-500/5',
        tone === 'bad' && 'border-destructive/30 bg-destructive/5',
      )}
    >
      <div className="text-xs uppercase tracking-wide text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
    </div>
  )
}

function buildComparison(exact: Experiment, approx: Experiment) {
  const rows: {
    label: string
    exact: string
    approx: string
    delta: string
    positive: boolean | null
  }[] = []

  const push = (
    label: string,
    exactValue: number | null | undefined,
    approxValue: number | null | undefined,
    format: (v: number) => string,
    higherIsBetter = true,
  ) => {
    if (exactValue == null || approxValue == null) return
    const diff = approxValue - exactValue
    rows.push({
      label,
      exact: format(exactValue),
      approx: format(approxValue),
      delta: `${diff > 0 ? '+' : ''}${format(diff).replace(/^\+/, '')}`,
      positive: higherIsBetter ? diff >= 0 : diff <= 0,
    })
  }

  push('Accuracy', exact.accuracy, approx.accuracy, (v) => `${(v * 100).toFixed(2)}%`)
  push('Macro F1', exact.macro_f1, approx.macro_f1, (v) => `${(v * 100).toFixed(2)}%`)
  push('Recall@K', exact.recall_at_k, approx.recall_at_k, (v) => `${(v * 100).toFixed(2)}%`)
  push(
    'Latency avg (ms)',
    exact.latency_avg_ms,
    approx.latency_avg_ms,
    (v) => v.toFixed(3),
    false,
  )
  push(
    'Latency p95 (ms)',
    exact.latency_p95_ms,
    approx.latency_p95_ms,
    (v) => v.toFixed(3),
    false,
  )

  return rows
}
