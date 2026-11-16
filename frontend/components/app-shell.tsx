import type { ReactNode } from 'react'

import { AppHeader } from '@/components/app-header'

export function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="flex min-h-screen flex-col">
      <AppHeader />
      <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6">
        {children}
      </main>
      <footer className="border-t border-border/70 py-6">
        <div className="mx-auto max-w-7xl px-4 text-xs text-muted-foreground sm:px-6">
          Master&apos;s ML Course Project — Design and Evaluation of a
          PostgreSQL-Based KNN Classification Platform Using Exact and
          Approximate Vector Search.
        </div>
      </footer>
    </div>
  )
}
