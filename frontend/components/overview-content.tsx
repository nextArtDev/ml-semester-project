'use client'

import { useQuery } from '@tanstack/react-query'
import {
  ArrowRight,
  Database,
  FlaskConical,
  Gauge,
  GitCompareArrows,
  Target,
} from 'lucide-react'
import { useTranslations } from 'next-intl'
import { Link } from '@/i18n/navigation'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { listExperiments } from '@/lib/api'
import { cn, formatMs, formatPercent } from '@/lib/utils'

function MetricCard({
  label,
  value,
  hint,
  icon: Icon,
  accent,
}: {
  label: string
  value: string
  hint?: string
  icon: typeof Target
  accent?: string
}) {
  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium text-muted-foreground">
          {label}
        </CardTitle>
        <Icon className={cn('size-4 text-muted-foreground', accent)} />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-semibold tabular-nums">{value}</div>
        {hint && (
          <p className="mt-1 text-xs text-muted-foreground">{hint}</p>
        )}
      </CardContent>
    </Card>
  )
}

export function OverviewContent() {
  const t = useTranslations('overview')
  const tCommon = useTranslations('common')
  const tExp = useTranslations('experiments')

  const { data: experiments, isLoading } = useQuery({
    queryKey: ['experiments'],
    queryFn: () => listExperiments(),
  })

  const completed = (experiments ?? []).filter((e) => e.status === 'completed')
  const exact = completed.find((e) => e.retrieval_method === 'exact')
  const approx = completed.find((e) => e.retrieval_method === 'hnsw')

  const speedup =
    exact?.latency_avg_ms && approx?.latency_avg_ms
      ? exact.latency_avg_ms / approx.latency_avg_ms
      : null

  const steps = [
    { icon: Database, key: 'step1', href: '/datasets' },
    { icon: FlaskConical, key: 'step2', href: '/experiments' },
    { icon: Gauge, key: 'step3', href: '/experiments' },
    { icon: GitCompareArrows, key: 'step4', href: '/compare' },
  ] as const

  return (
    <>
      <Card className="border-primary/20 bg-primary/[0.03]">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-base">
            <Target className="size-4 text-primary" />
            {t('question')}
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="max-w-4xl text-sm leading-relaxed text-foreground/85">
            {t('questionBody')}
          </p>
        </CardContent>
      </Card>

      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
        {isLoading ? (
          Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-[7.5rem] w-full" />
          ))
        ) : (
          <>
            <MetricCard
              label={t('accuracy')}
              value={formatPercent(exact?.accuracy)}
              icon={Target}
            />
            <MetricCard
              label={t('macroF1')}
              value={formatPercent(exact?.macro_f1)}
              icon={Target}
            />
            <MetricCard
              label={t('recallAtK')}
              value={formatPercent(approx?.recall_at_k)}
              icon={Gauge}
            />
            <MetricCard
              label={t('latencyP95')}
              value={formatMs(exact?.latency_p95_ms)}
              icon={Gauge}
            />
            <MetricCard
              label={t('speedup')}
              value={speedup ? `${speedup.toFixed(2)}×` : '—'}
              hint={approx ? `ef_search = ${approx.hnsw_ef_search ?? '—'}` : undefined}
              icon={Gauge}
              accent="text-emerald-500"
            />
          </>
        )}
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('quickStart')}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {steps.map((step, i) => (
              <Link
                key={step.key}
                href={step.href}
                className="group flex items-start gap-3 rounded-lg border border-transparent p-2.5 transition-colors hover:border-border hover:bg-secondary/50"
              >
                <span className="mt-0.5 grid size-7 shrink-0 place-items-center rounded-full bg-secondary text-xs font-semibold text-secondary-foreground">
                  {i + 1}
                </span>
                <span className="flex-1 text-sm text-muted-foreground group-hover:text-foreground">
                  {t(step.key)}
                </span>
                <step.icon className="mt-0.5 size-4 shrink-0 text-muted-foreground/60" />
              </Link>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('recent')}</CardTitle>
            <CardDescription>{t('title')}</CardDescription>
          </CardHeader>
          <CardContent>
            {completed.length === 0 ? (
              <p className="text-sm text-muted-foreground">{t('noData')}</p>
            ) : (
              <ul className="flex flex-col gap-2.5">
                {completed.slice(0, 6).map((exp) => (
                  <li
                    key={exp.id}
                    className="flex items-center justify-between gap-3 rounded-lg border border-border/70 p-3"
                  >
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <Badge
                          variant={
                            exp.retrieval_method === 'hnsw' ? 'default' : 'secondary'
                          }
                          className="shrink-0"
                        >
                          {exp.retrieval_method === 'hnsw'
                            ? tCommon('approximate')
                            : tCommon('exact')}
                        </Badge>
                        <span className="truncate text-sm font-medium">
                          {exp.name ?? `#${exp.id}`}
                        </span>
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground">
                        k={exp.k} · {exp.distance_metric.toUpperCase()} ·{' '}
                        {tExp('recallAtK')} {formatPercent(exp.recall_at_k)} ·{' '}
                        {formatMs(exp.latency_p50_ms)}
                      </p>
                    </div>
                    <div className="shrink-0 text-end">
                      <div className="text-sm font-semibold tabular-nums">
                        {formatPercent(exp.accuracy)}
                      </div>
                      <div className="text-[11px] text-muted-foreground">
                        {tExp('accuracy')}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </CardContent>
        </Card>
      </section>

      <div className="flex justify-end">
        <Button asChild variant="outline">
          <Link href="/methodology">
            {tCommon('all')}
            <ArrowRight className="size-4 rtl:rotate-180" />
          </Link>
        </Button>
      </div>
    </>
  )
}
