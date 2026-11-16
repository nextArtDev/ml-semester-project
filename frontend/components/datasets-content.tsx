'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Database, FileUp, Loader2, Trash2 } from 'lucide-react'
import { useTranslations } from 'next-intl'
import { useRef, useState } from 'react'
import { toast } from 'sonner'

import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { deleteDataset, ingestCovertype, listDatasets, uploadDataset } from '@/lib/api'
import { formatNumber } from '@/lib/utils'

export function DatasetsContent() {
  const t = useTranslations('datasets')
  const tCommon = useTranslations('common')
  const queryClient = useQueryClient()
  const fileInputRef = useRef<HTMLInputElement>(null)

  const [covertypeName, setCovertypeName] = useState('Forest CoverType')
  const [maxRows, setMaxRows] = useState('')
  const [maxTrainRows, setMaxTrainRows] = useState('')
  const [testSize, setTestSize] = useState('5000')
  const [uploadName, setUploadName] = useState('')
  const [targetColumn, setTargetColumn] = useState('label')
  const [pendingFile, setPendingFile] = useState<File | null>(null)

  const { data: datasets, isLoading } = useQuery({
    queryKey: ['datasets'],
    queryFn: listDatasets,
  })

  const invalidate = () => queryClient.invalidateQueries({ queryKey: ['datasets'] })

  const ingestMutation = useMutation({
    mutationFn: () =>
      ingestCovertype({
        name: covertypeName || 'Forest CoverType',
        max_rows: maxRows ? Number(maxRows) : null,
        max_train_rows: maxTrainRows ? Number(maxTrainRows) : null,
        test_size: testSize ? Number(testSize) : null,
      }),
    onSuccess: (result) => {
      toast.success(
        `${result.name}: ${formatNumber(result.n_train)} train / ${formatNumber(result.n_test)} test`,
      )
      invalidate()
    },
    onError: (error: Error) => toast.error(error.message),
  })

  const uploadMutation = useMutation({
    mutationFn: async () => {
      if (!pendingFile) throw new Error('No file selected')
      const form = new FormData()
      form.append('file', pendingFile)
      form.append('name', uploadName || pendingFile.name.replace(/\.csv$/i, ''))
      form.append('target_column', targetColumn)
      if (testSize) form.append('test_size', testSize)
      return uploadDataset(form)
    },
    onSuccess: (result) => {
      toast.success(
        `${result.name}: ${formatNumber(result.n_train)} train / ${formatNumber(result.n_test)} test`,
      )
      setPendingFile(null)
      invalidate()
    },
    onError: (error: Error) => toast.error(error.message),
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDataset,
    onSuccess: () => {
      toast.success(t('delete'))
      invalidate()
    },
    onError: (error: Error) => toast.error(error.message),
  })

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-3">
        <h1 className="text-3xl font-semibold tracking-tight">{t('title')}</h1>
        <p className="max-w-3xl text-muted-foreground">{t('subtitle')}</p>
      </div>

      <Tabs defaultValue="covertype">
        <TabsList>
          <TabsTrigger value="covertype" className="gap-2">
            <Database className="size-4" />
            {t('ingestCovertype')}
          </TabsTrigger>
          <TabsTrigger value="upload" className="gap-2">
            <FileUp className="size-4" />
            {t('upload')}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="covertype">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t('ingestCovertype')}</CardTitle>
              <CardDescription>
                UCI Forest CoverType â€” 581,012 rows, 54 features, 7 classes.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
              <div className="grid gap-2 lg:col-span-2">
                <Label htmlFor="ct-name">{t('name')}</Label>
                <Input
                  id="ct-name"
                  value={covertypeName}
                  onChange={(e) => setCovertypeName(e.target.value)}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="ct-maxrows">{t('maxRows')}</Label>
                <Input
                  id="ct-maxrows"
                  type="number"
                  placeholder="581012"
                  value={maxRows}
                  onChange={(e) => setMaxRows(e.target.value)}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="ct-maxtrain">{t('maxTrainRows')}</Label>
                <Input
                  id="ct-maxtrain"
                  type="number"
                  placeholder="581012"
                  value={maxTrainRows}
                  onChange={(e) => setMaxTrainRows(e.target.value)}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="ct-test">{t('testSize')}</Label>
                <Input
                  id="ct-test"
                  type="number"
                  value={testSize}
                  onChange={(e) => setTestSize(e.target.value)}
                />
              </div>

              <div className="sm:col-span-2 lg:col-span-5">
                <Button
                  onClick={() => ingestMutation.mutate()}
                  disabled={ingestMutation.isPending}
                >
                  {ingestMutation.isPending && (
                    <Loader2 className="size-4 animate-spin" />
                  )}
                  {ingestMutation.isPending ? t('ingesting') : t('ingest')}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="upload">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">{t('upload')}</CardTitle>
              <CardDescription>{t('uploadHint')}</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <div className="grid gap-2 sm:col-span-2 lg:col-span-3">
                <Label>{t('dropFile')}</Label>
                <div
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(e) => e.preventDefault()}
                  onDrop={(e) => {
                    e.preventDefault()
                    const file = e.dataTransfer.files?.[0]
                    if (file) setPendingFile(file)
                  }}
                  className="cursor-pointer rounded-lg border border-dashed border-border bg-muted/30 p-8 text-center text-sm text-muted-foreground transition-colors hover:border-primary/50 hover:bg-muted/50"
                >
                  {pendingFile ? (
                    <span className="font-medium text-foreground">
                      {pendingFile.name}
                    </span>
                  ) : (
                    t('dropFile')
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,text/csv"
                    className="hidden"
                    onChange={(e) => setPendingFile(e.target.files?.[0] ?? null)}
                  />
                </div>
              </div>

              <div className="grid gap-2">
                <Label htmlFor="up-name">{t('name')}</Label>
                <Input
                  id="up-name"
                  value={uploadName}
                  onChange={(e) => setUploadName(e.target.value)}
                  placeholder={pendingFile?.name}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="up-target">{t('targetColumn')}</Label>
                <Input
                  id="up-target"
                  value={targetColumn}
                  onChange={(e) => setTargetColumn(e.target.value)}
                />
              </div>
              <div className="grid gap-2">
                <Label htmlFor="up-test">{t('testSize')}</Label>
                <Input
                  id="up-test"
                  type="number"
                  value={testSize}
                  onChange={(e) => setTestSize(e.target.value)}
                />
              </div>

              <div className="sm:col-span-2 lg:col-span-3">
                <Button
                  onClick={() => uploadMutation.mutate()}
                  disabled={!pendingFile || uploadMutation.isPending}
                >
                  {uploadMutation.isPending && (
                    <Loader2 className="size-4 animate-spin" />
                  )}
                  {t('upload')}
                </Button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">{t('title')}</CardTitle>
        </CardHeader>
        <CardContent>
          {isLoading ? (
            <p className="text-sm text-muted-foreground">{tCommon('loading')}</p>
          ) : !datasets || datasets.length === 0 ? (
            <p className="text-sm text-muted-foreground">{t('empty')}</p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>#</TableHead>
                  <TableHead>{t('name')}</TableHead>
                  <TableHead>{t('source')}</TableHead>
                  <TableHead className="text-end">{t('rows')}</TableHead>
                  <TableHead className="text-end">{t('features')}</TableHead>
                  <TableHead className="text-end">{t('classes')}</TableHead>
                  <TableHead className="text-end">{t('dimension')}</TableHead>
                  <TableHead className="w-12" />
                </TableRow>
              </TableHeader>
              <TableBody>
                {datasets.map((ds) => (
                  <TableRow key={ds.id}>
                    <TableCell className="text-muted-foreground tabular-nums">
                      {ds.id}
                    </TableCell>
                    <TableCell className="font-medium">{ds.name}</TableCell>
                    <TableCell>
                      <Badge variant="secondary">{ds.source}</Badge>
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {formatNumber(ds.n_rows)}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {ds.n_features}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {ds.n_classes}
                    </TableCell>
                    <TableCell className="text-end tabular-nums">
                      {ds.dimension}
                    </TableCell>
                    <TableCell>
                      <Button
                        variant="ghost"
                        size="icon"
                        aria-label={tCommon('delete')}
                        onClick={() => deleteMutation.mutate(ds.id)}
                      >
                        <Trash2 className="size-4 text-destructive/70" />
                      </Button>
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
