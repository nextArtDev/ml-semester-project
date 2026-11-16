'use client'

import { useLocale, useTranslations } from 'next-intl'
import { useTransition } from 'react'
import { Languages, Moon, Sun } from 'lucide-react'
import { useTheme } from 'next-themes'

import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Link, usePathname, useRouter } from '@/i18n/navigation'
import { cn } from '@/lib/utils'

const NAV_ITEMS = [
  { href: '/', key: 'overview' },
  { href: '/datasets', key: 'datasets' },
  { href: '/experiments', key: 'experiments' },
  { href: '/compare', key: 'compare' },
  { href: '/methodology', key: 'methodology' },
] as const

export function AppHeader() {
  const t = useTranslations('nav')
  const locale = useLocale()
  const pathname = usePathname()
  const router = useRouter()
  const { setTheme, resolvedTheme } = useTheme()
  const [isPending, startTransition] = useTransition()

  // The locale-aware router from next-intl swaps the locale segment while
  // keeping the current route, so no full page reload is needed.
  const switchLocale = (next: string) => {
    startTransition(() => {
      router.replace(pathname, { locale: next })
    })
  }

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border/70 bg-background/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center gap-4 px-4 sm:px-6">
        <Link href="/" className="flex shrink-0 items-center gap-2.5">
          <span className="grid size-9 place-items-center rounded-lg bg-primary text-primary-foreground">
            <svg viewBox="0 0 24 24" className="size-5" fill="none" aria-hidden>
              <circle cx="6" cy="7" r="2.4" fill="currentColor" opacity="0.9" />
              <circle cx="13" cy="5" r="1.8" fill="currentColor" opacity="0.6" />
              <circle cx="12" cy="13" r="2.6" fill="currentColor" opacity="0.8" />
              <circle cx="18" cy="17" r="1.8" fill="currentColor" opacity="0.5" />
              <path
                d="M6 7 L12 13 M13 5 L12 13 M12 13 L18 17"
                stroke="currentColor"
                strokeWidth="1.2"
                opacity="0.55"
              />
            </svg>
          </span>
          <span className="hidden flex-col leading-tight sm:flex">
            <span className="text-sm font-semibold tracking-tight">KNN Platform</span>
            <span className="text-[11px] text-muted-foreground">
              PostgreSQL · pgvector
            </span>
          </span>
        </Link>

        <nav className="flex min-w-0 flex-1 items-center gap-1 overflow-x-auto">
          {NAV_ITEMS.map((item) => {
            const active =
              item.href === '/'
                ? pathname === '/'
                : pathname.startsWith(item.href)
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  'whitespace-nowrap rounded-md px-3 py-1.5 text-sm font-medium transition-colors',
                  active
                    ? 'bg-secondary text-foreground'
                    : 'text-muted-foreground hover:bg-secondary/60 hover:text-foreground',
                )}
              >
                {t(item.key)}
              </Link>
            )
          })}
        </nav>

        <div className="flex shrink-0 items-center gap-1">
          <Button
            variant="ghost"
            size="icon"
            aria-label="Toggle theme"
            onClick={() => setTheme(resolvedTheme === 'dark' ? 'light' : 'dark')}
          >
            <Sun className="size-4 dark:hidden" />
            <Moon className="hidden size-4 dark:block" />
          </Button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" size="icon" aria-label={t('language')}>
                <Languages className="size-4" />
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end">
              <DropdownMenuItem
                disabled={isPending}
                onSelect={() => switchLocale('en')}
                className={cn(locale === 'en' && 'bg-secondary')}
              >
                English
              </DropdownMenuItem>
              <DropdownMenuItem
                disabled={isPending}
                onSelect={() => switchLocale('fa')}
                className={cn(locale === 'fa' && 'bg-secondary')}
              >
                فارسی
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>
    </header>
  )
}
