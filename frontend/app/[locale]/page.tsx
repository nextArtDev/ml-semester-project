import { getTranslations, setRequestLocale } from 'next-intl/server'

import { AppShell } from '@/components/app-shell'
import { HealthBadge } from '@/components/health-badge'
import { OverviewContent } from '@/components/overview-content'

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>
}) {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'overview' })
  return { title: t('title') }
}

export default async function OverviewPage({
  params,
}: {
  params: Promise<{ locale: string }>
}) {
  const { locale } = await params
  setRequestLocale(locale)
  const t = await getTranslations('overview')

  return (
    <AppShell>
      <div className="flex flex-col gap-8">
        <section className="flex flex-col gap-3">
          <h1 className="text-3xl font-semibold tracking-tight sm:text-4xl">
            {t('title')}
          </h1>
          <p className="max-w-3xl text-muted-foreground">{t('subtitle')}</p>
          <HealthBadge />
        </section>

        <OverviewContent />
      </div>
    </AppShell>
  )
}
