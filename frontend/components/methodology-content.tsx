'use client'

import {
  Database,
  FileCode2,
  Info,
  Layers3,
  Server,
  ShieldCheck,
} from 'lucide-react'
import { useTranslations } from 'next-intl'

import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'

export function MethodologyContent() {
  const t = useTranslations('methodology')

  const architecture = [
    { icon: FileCode2, title: t('archFrontend'), detail: 'Next.js 16 Â· TypeScript Â· Tailwind v4 Â· Recharts Â· next-intl' },
    { icon: Server, title: t('archBackend'), detail: 'Python 3.12 Â· FastAPI Â· psycopg 3 Â· scikit-learn Â· pandas' },
    { icon: Database, title: t('archDatabase'), detail: 'PostgreSQL 18 Â· pgvector 0.8.7 Â· HNSW index' },
  ]

  const pipeline = [
    t('step1'),
    t('step2'),
    t('step3'),
    t('step4'),
    t('step5'),
    t('step6'),
  ]

  const metrics = [
    { icon: ShieldCheck, text: t('metricClassification') },
    { icon: Layers3, text: t('metricRetrieval') },
    { icon: Info, text: t('metricLatency') },
    { icon: Database, text: t('metricIndex') },
  ]

  const notes = [t('noteExact'), t('noteIsolation'), t('noteWarmup')]

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <h1 className="text-3xl font-semibold tracking-tight">{t('title')}</h1>
        <p className="max-w-3xl text-muted-foreground">{t('subtitle')}</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t('architecture')}</CardTitle>
        </CardHeader>
        <CardContent className="grid gap-4 md:grid-cols-3">
          {architecture.map((item) => (
            <div
              key={item.title}
              className="rounded-lg border border-border/70 bg-muted/25 p-4"
            >
              <item.icon className="size-5 text-primary" />
              <div className="mt-3 text-sm font-semibold">{item.title}</div>
              <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                {item.detail}
              </p>
            </div>
          ))}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t('pipeline')}</CardTitle>
        </CardHeader>
        <CardContent>
          <ol className="flex flex-col gap-3">
            {pipeline.map((step, i) => (
              <li key={step} className="flex items-start gap-3">
                <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full bg-primary/10 text-xs font-semibold text-primary">
                  {i + 1}
                </span>
                <span className="text-sm text-muted-foreground">{step}</span>
              </li>
            ))}
          </ol>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('metrics')}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {metrics.map((item) => (
              <div key={item.text} className="flex items-start gap-3">
                <item.icon className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">{item.text}</span>
              </div>
            ))}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">{t('notes')}</CardTitle>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {notes.map((note) => (
              <div key={note} className="flex items-start gap-3">
                <Info className="mt-0.5 size-4 shrink-0 text-muted-foreground" />
                <span className="text-sm text-muted-foreground">{note}</span>
              </div>
            ))}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
