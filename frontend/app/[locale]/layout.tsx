import type { Metadata } from 'next'
import localFont from 'next/font/local'
import { NextIntlClientProvider, getMessages } from 'next-intl'
import { getTranslations, setRequestLocale } from 'next-intl/server'
import { Toaster } from 'sonner'

import '../globals.css'
import { QueryProviders } from '@/providers/query-provider'
import { ThemeProvider } from '@/providers/theme-providers'
import { routing } from '@/i18n/routing'

const interFont = localFont({
  src: '../public/fonts/Inter-Variable.woff2',
  variable: '--font-sans-ui',
  display: 'swap',
  fallback: ['system-ui', 'Segoe UI', 'Helvetica Neue', 'Arial', 'sans-serif'],
})

const vazirFont = localFont({
  src: '../public/fonts/vazirmatn.woff2',
  variable: '--font-fa-ui',
  display: 'swap',
  fallback: ['Tahoma', 'system-ui', 'sans-serif'],
})

export function generateStaticParams() {
  return routing.locales.map((locale) => ({ locale }))
}

export async function generateMetadata({
  params,
}: {
  params: Promise<{ locale: string }>
}): Promise<Metadata> {
  const { locale } = await params
  const t = await getTranslations({ locale, namespace: 'metadata' })

  return {
    title: { default: t('title'), template: `%s — ${t('title')}` },
    description: t('description'),
  }
}

export default async function LocaleLayout({
  children,
  params,
}: {
  children: React.ReactNode
  params: Promise<{ locale: string }>
}) {
  const { locale } = await params
  setRequestLocale(locale)

  const messages = await getMessages()
  const isRtl = locale === 'fa'

  return (
    <html
      lang={locale}
      dir={isRtl ? 'rtl' : 'ltr'}
      suppressHydrationWarning
      className={`${interFont.variable} ${vazirFont.variable}`}
    >
      <body className="min-h-screen antialiased">
        <NextIntlClientProvider messages={messages}>
          <QueryProviders>
            <ThemeProvider
              attribute="class"
              defaultTheme="light"
              enableSystem
              disableTransitionOnChange
            >
              {children}
              <Toaster richColors position="bottom-right" />
            </ThemeProvider>
          </QueryProviders>
        </NextIntlClientProvider>
      </body>
    </html>
  )
}
