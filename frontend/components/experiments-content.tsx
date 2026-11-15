'use client'

import { useQuery } from '@tanstack/react-query'
import { useTranslations } from 'next-intl'
import { useState } from 'react'

import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { ExperimentResultPanel } from '@/components/experiment-result-panel'
import { ExperimentRunner } from '@/components/experiment-runner'
import { listExperiments } from '@/lib/api'
import { formatMs, formatPercent } from '@/lib/utils'

export function ExperimentsContent() {
  const t = useTranslations('experiments')
  const tCommon = useTranslations('common')

  const [datasetId, setDatasetId] = useState<number | undefined>(undefined)
  const [selectedId, setSelectedId] = useState<number | undefined>(undefined)

  const { data: experiments } = useQuery({
    queryKey: ['experiments', datasetId],
    queryFn: () => listExperiments(datasetId),
  })

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <h1 className="text-3xl font-semibold tracking-tight">{t('title')}</h1>
        <p className="max-w-3xl text-muted-foreground">{t('subtitle')}</p>
      </div>

      <ExperimentRunner
        datasetId={datasetId}
        onDatasetChange={(id) => {
          setDatasetId(id)
          setSelectedId(undefined)
        }}
      />

      <ExperimentResultPanel experimentId={selectedId} />

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t('history')}</CardTitle>
          <CardDescription>{t('subtitle')}</CardDescription>
        </CardHeader>
        <CardContent>
          {!experiments || experiments.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t('noExperiments')}</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>#</TableHead>
                  <TableHead>{tCommon('name')}</TableHead>
                  <TableHead>{tCommon('status')}</TableHead>
                  <TableHead className="text-end">k</TableHead>
                  <TableHead className="text-end">ef</TableHead>
                  <TableHead className="text-end">{t('accuracy')}</TableHead>
                  <TableHead className="text-end">{t('recallAtK')}</TableHead>
                  <TableHead className="text-end">{t('latencyP50')}</TableHead>
                  <TableHead className="text-end">{t('latencyP95')}</TableHead>
                  <TableHead className="w-24" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {experiments.map((exp) => (
                  <TableRow
                    key={exp.id}
                    className={
                      selectedId === exp.id ? 'bg-secondary/50' : undefined
                    }
                  >
                    <TableCell className="tabular-nums text-muted-foreground">
                      {exp.id}
                    </TableCell>
                    <TableCell className="font-medium">
                      {exp.name ?? '—'}
                    </TableCell>
                    <TableCell>
                      <Badge
                        variant={
                          exp.retrieval_method === 'hnsw' ? 'default' : 'secondary'
                        }
                      >
                        {exp.retrieval_method === 'hnsw'
                          ? tCommon('approximate')
                          : tCommon('exact')}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-end tabular-nums">{exp.k}</TableCell>
                    <TableCell className="text-end tabular-nums">
                      {exp.hnsw_ef_search ?? '—'}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {formatPercent(exp.accuracy)}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {formatPercent(exp.recall_at_k)}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {formatMs(exp.latency_p50_ms)}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {formatMs(exp.latency_p95_ms)}
                    </TableCell>
                    <TableCell>
                      <button
                        type="button"
                        onClick={() => setSelectedId(exp.id)}
                        className="text-xs font-medium text-primary hover:underline"
                      >
                        {t('viewResults')}
                      </button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
