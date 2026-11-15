'use client'

import { useQuery } from '@tanstack/react-query'
import { Activity, Database, Layers, Server } from 'lucide-react'
import { useTranslations } from 'next-intl'

import { getHealth } from '@/lib/api'
import { cn } from '@/lib/utils'

export function HealthBadge() {
  const t = useTranslations('health')
  const { data, isLoading } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30_000,
  })

  if (isLoading) {
    return (
      <div className="h-9 w-40 animate-pulse rounded-md bg-muted" />
    )
  }

  const connected = data?.database === 'connected'

  return (
    <div className="flex flex-wrap items-center gap-2">
      <span
        className={cn(
          'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium',
          connected
            ? 'border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-400'
            : 'border-destructive/30 bg-destructive/10 text-destructive',
        )}
      >
        <span className="relative flex size-1.5">
          <span
            className={cn(
              'absolute inline-flex size-full animate-ping rounded-full opacity-70',
              connected ? 'bg-emerald-500' : 'bg-destructive',
            )}
          />
          <span
            className={cn(
              'relative inline-flex size-1.5 rounded-full',
              connected ? 'bg-emerald-500' : 'bg-destructive',
            )}
          />
        </span>
        <Database className="size-3" />
        {connected ? t('connected') : t('disconnected')}
      </span>

      {data?.pgvector_version && (
        <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/50 px-2.5 py-1 text-xs text-muted-foreground">
          <Layers className="size-3" />
          {t('pgvector')} {data.pgvector_version}
        </span>
      )}

      <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/50 px-2.5 py-1 text-xs text-muted-foreground">
        <Server className="size-3" />
        {data?.dataset_count ?? 0} · {data?.experiment_count ?? 0}
      </span>
    </div>
  )
}
