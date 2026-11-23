# Phase 4 — Scientific Analysis

**Sources used:** `docs/results/validated/summary.csv`, `docs/results/validated/statistical_summary.json`, `docs/reports/PHASE3-BENCHMARK-REPORT.md`, `docs/reports/PHASE3-RESULTS-VALIDATION.md`, `docs/reports/PHASE3-BENCHMARK-PROTOCOL.md`  
**Raw measurements modified:** none  
**Natural / controlled modes:** kept separate throughout  
**Scope:** Covertype-derived pools of 10,000 and 50,000 training rows, L2 distance, k=5, warm cache, PostgreSQL 18.0 / pgvector 0.8.7

---

## 1. Reproducibility

All figures and tables were generated from the validated outputs by:

```bash
cd backend
.venv/Scripts/python.exe ../scripts/generate_research_figures.py
```

Outputs:

- `docs/results/figures/figure_01_latency_profiles.png` … `figure_08_macro_precision_recall.png`
- `docs/results/tables/main_benchmark_comparison.csv`
- `docs/results/tables/per_configuration_metrics.csv`
- `docs/results/tables/statistical_comparison.csv`
- `docs/results/tables/index_construction_storage.csv`
- `docs/results/tables/per_class_results.csv` (explicit availability notice; see limitations)
- `docs/results/tables/summary_of_research_findings.csv`

## 2. Figures

| Figure | Caption |
|---|---|
| 1 | **Latency profiles by configuration.** Lines span validated min–max latency; markers show p50, p95 and p99. Top: natural planner. Bottom: controlled HNSW. These are validated percentile summaries, not raw distribution plots. |
| 2 | **Recall@5 versus p50 latency.** Each point is a validated configuration; crosses are controlled HNSW diagnostics and are not pooled with natural results. |
| 3 | **Recall@5 sensitivity to ef_search.** Natural 10k/ef=400 is absent because it failed the access-path check. |
| 4 | **Classification accuracy by validated configuration.** Exact is the same-pool reference; natural planner only. |
| 5 | **Macro-F1 by validated configuration.** The gap from accuracy reflects class imbalance and per-class support. |
| 6 | **Training-pool size versus p50 query latency.** Exact latency grows strongly; HNSW remains comparatively flat. |
| 7 | **HNSW index storage and first-build cost.** Reused indexes have no fresh build time in these runs. |
| 8 | **Macro precision–recall by validated configuration.** This is the available class-level proxy; true per-class matrices were not exported to validated outputs. |

## 3. Measured observations

### 3.1 Main benchmark comparison

Averages over the three valid rounds per configuration (natural planner only):

| Pool | Configuration | Recall@5 | Neighbour Jaccard | Accuracy | Macro F1 | p50 (ms) | p95 (ms) | p50 speed-up |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | Exact | 1.0000 | 1.0000 | 0.7547 | 0.5605 | 2.386 | 2.765 | — |
| 10,000 | HNSW ef=10 | 0.9949 | 0.9931 | 0.7537 | 0.5588 | 0.298 | 0.398 | 8.01× |
| 10,000 | HNSW ef=50 | 0.9998 | 0.9997 | 0.7550 | 0.5606 | 0.431 | 0.579 | 5.56× |
| 10,000 | HNSW ef=100 | 0.9999 | 0.9999 | 0.7547 | 0.5605 | 0.582 | 0.798 | 4.11× |
| 50,000 | Exact | 1.0000 | 1.0000 | 0.8440 | 0.7500 | 11.278 | 11.786 | — |
| 50,000 | HNSW ef=10 | 0.9829 | 0.9814 | 0.8420 | 0.7460 | 0.285 | 0.361 | 39.63× |
| 50,000 | HNSW ef=50 | 0.9975 | 0.9970 | 0.8433 | 0.7498 | 0.416 | 0.539 | 27.17× |
| 50,000 | HNSW ef=100 | 0.9988 | 0.9984 | 0.8437 | 0.7499 | 0.568 | 0.764 | 19.94× |
| 50,000 | HNSW ef=400 | 0.9996 | 0.9994 | 0.8440 | 0.7500 | 1.381 | 1.805 | 8.17× |

The 10,000/ef=400 natural cell is absent: all three rounds were sequential-scan access-path failures and were excluded from validated HNSW summaries.

### 3.2 Latency comparisons and statistical evidence

Paired per-query latency differences (`HNSW − Exact`) and Holm-adjusted Wilcoxon results:

| Pool | ef | Median Δ (ms) | 95% bootstrap CI (ms) | Holm p | Rank-biserial |
|---:|---:|---:|---|---:|---:|
| 10,000 | 10 | −2.043 | [−2.050, −2.030] | <1e-300 | −1.000 |
| 10,000 | 50 | −1.935 | [−1.944, −1.926] | <1e-300 | −1.000 |
| 10,000 | 100 | −1.731 | [−1.740, −1.720] | <1e-300 | −1.000 |
| 50,000 | 10 | −10.927 | [−10.932, −10.923] | <1e-300 | −1.000 |
| 50,000 | 50 | −10.803 | [−10.808, −10.797] | <1e-300 | −1.000 |
| 50,000 | 100 | −10.654 | [−10.660, −10.647] | <1e-300 | −1.000 |
| 50,000 | 400 | −9.798 | [−9.808, −9.789] | <1e-300 | −1.000 |

Wilcoxon p-values underflowed to `0.0` in double precision and are reported as `<1e-300`. Rank-biserial `−1` means the aggregated latency of every paired query favoured HNSW. This is a distributional/ordering result, not proof of practical superiority.

### 3.3 Retrieval accuracy and classification

Mean Recall@5 differences versus exact:

| Pool | ef | Mean Δ Recall@5 | 95% CI | Δ accuracy | McNemar Holm p |
|---:|---:|---:|---|---:|---:|
| 10,000 | 10 | −0.00507 | [−0.00700, −0.00340] | −0.0010 | 1.000 |
| 10,000 | 50 | −0.00020 | [−0.00047, 0.00000] | +0.0003 | 1.000 |
| 10,000 | 100 | −0.00007 | [−0.00020, 0.00000] | 0.0000 | 1.000 |
| 50,000 | 10 | −0.01707 | [−0.02154, −0.01280] | −0.0020 | 1.000 |
| 50,000 | 50 | −0.00253 | [−0.00427, −0.00107] | −0.0007 | 1.000 |
| 50,000 | 100 | −0.00120 | [−0.00220, −0.00040] | −0.0003 | 1.000 |
| 50,000 | 400 | −0.00040 | [−0.00080, −0.00007] | 0.0000 | 1.000 |

No classification accuracy difference is statistically significant after Holm correction.

Descriptive Pearson correlations across the ten aggregated configurations:

- Recall@5 vs accuracy pooled across both sizes: **−0.263**.
- Within 10k: **+0.953**; within 50k: **+0.978**.

The pooled sign reversal is a size-confounding artifact: the 50k pool has higher baseline accuracy despite lower recall at ef=10. These correlations are descriptive; each size has only four or five configurations.

### 3.4 ef_search sensitivity

- **10k:** valid natural ef=10/50/100 improved Recall@5 from 0.9949 → 0.9998 → 0.9999 while p50 latency rose from 0.298 → 0.431 → 0.582 ms.
- **50k:** ef=10/50/100/400 improved Recall@5 from 0.9829 → 0.9975 → 0.9988 → 0.9996 while p50 latency rose from 0.285 → 0.416 → 0.568 → 1.381 ms.
- The largest recall gain is from ef=10 to ef=50; gains after ef=100 are small.
- 10k/ef=400 cannot be assessed in natural mode because the planner chose a sequential scan.

### 3.5 Training-size sensitivity

- Exact p50 latency rose from 2.386 ms (10k) to 11.278 ms (50k), about 4.7×.
- HNSW p50 stayed nearly flat for comparable ef values: 0.298 → 0.285 ms (ef=10), 0.431 → 0.416 ms (ef=50), 0.582 → 0.568 ms (ef=100).
- Consequently, p50 speed-ups widened from 4.11–8.01× at 10k to 19.94–39.63× at 50k.
- Accuracy rose with pool size for both methods: exact 0.7547 → 0.8440; HNSW ef=50 0.7550 → 0.8433.

### 3.6 Index construction and storage

| Pool | Fresh build time | Index size |
|---:|---:|---:|
| 10,000 | 1,478.5 ms | 5,226,496 B (4.984 MB) |
| 50,000 | 3,500.2 ms | 26,083,328 B (24.875 MB) |

Each build time appears once; other ef values reused the same index. The index is therefore an up-front cost paid per pool, not per ef setting.

### 3.7 Controlled HNSW diagnostic

All eight controlled runs used the HNSW index and completed. They are reported separately from natural results. At 10k, controlled ef=400 produced a valid index scan where the natural planner failed; this demonstrates planner sensitivity, not superiority of forced execution under natural conditions.

## 4. Interpretations

### RQ1 — How much faster is HNSW?

**Interpretation:** In this setup, HNSW is substantially faster. The advantage is modest in absolute milliseconds at 10k but large at 50k, where paired medians are roughly 10 ms lower and relative speed-ups reach ~40× at ef=10. The consistency of paired ordering (rank-biserial −1) is strong evidence that the latency advantage is not driven by a few queries.

### RQ2 — What retrieval accuracy is sacrificed?

**Interpretation:** The sacrifice is small but measurable. It is negligible at higher ef on 10k and at ef≥50 on 50k. ef=10 on 50k is the clear outlier, losing about 1.7 Recall@5 points. Exact remains the only method with perfect identifier-level neighbour agreement.

### RQ3 — Does retrieval recall correlate with classification performance?

**Interpretation:** Within a pool size, higher recall tends to coincide with higher accuracy, but the classification differences are not statistically significant after Holm correction. The pooled negative correlation shows that recall is not a sufficient proxy for classification quality across different training sizes. KNN voting is robust to small neighbour-set differences when the retrieved neighbours share labels.

### RQ4 — How does ef_search influence recall and latency?

**Interpretation:** ef_search controls a clear recall–latency trade-off. Diminishing returns appear after ef=50/100. The choice of ef should be driven by the acceptable recall loss and latency budget for a specific pool size; ef=10 is not automatically the best latency option when classification quality matters.

### RQ5 — How does training size affect the trade-off?

**Interpretation:** The benefit of indexing grows with pool size. Exact scan cost scales with the table, while HNSW latency is comparatively stable. For small pools, lower absolute latency advantages must be weighed against index build and storage cost.

### RQ6 — What is the index construction and storage overhead?

**Interpretation:** Build time and storage are modest and paid once per pool. The 50k index adds about 25 MB and 3.5 s of construction time. Reuse across ef settings is essential to the fairness of the sweep.

### RQ7 — Does natural planner selection differ from controlled HNSW?

**Interpretation:** Yes. The natural planner rejects the 10k/ef=400 index because its cost estimate exceeds a sequential scan. Controlled execution obtains an index scan, but that changes the execution conditions and is not equivalent to natural planner behavior. Reporting the modes separately is scientifically necessary.

### RQ8 — When is Exact KNN preferable?

**Interpretation:** Exact is preferable when deterministic full neighbour recall matters, when no index overhead is acceptable, or when a pool is small enough that absolute latency differences are minor. It is also preferable when planner instability would undermine reproducibility. Its disadvantage grows sharply with pool size.

### RQ9 — When is HNSW preferable?

**Interpretation:** HNSW is preferable for larger pools with warm-cache latency requirements, provided the measured recall loss is acceptable and ef is tuned. On this dataset, ef=50 or 100 offers near-exact recall with much lower p50 latency at 50k. These conclusions are specific to Covertype, L2, k=5 and this machine.

## 5. Unsupported conclusions and limitations

- **No universal superiority claim:** one dataset family, metric, k and machine were tested.
- **No full latency distributions:** validated outputs contain percentiles and extrema, not raw per-query latency arrays; Figure 1 is a percentile profile.
- **Per-class results unavailable:** Prompt 6 recomputed per-class metrics and confusion matrices but did not export them to `docs/results/validated/`. `tables/per_class_results.csv` therefore documents non-availability; no per-class or confusion-matrix figure is fabricated.
- **Failed natural cell:** 10k/ef=400 has no valid natural HNSW measurement; it is not replaced by controlled results.
- **Controlled mode is descriptive:** it must not be merged into natural summaries.
- **Warm cache only:** cold-cache behavior is unmeasured.
- **Client-side latency:** includes Python/psycopg round trip, not pure server execution.
- **Limited replication:** three rounds and three repetitions; bootstrap intervals describe this query set and configuration, not all possible queries.
- **Repeated passes are correlated:** they were aggregated per query rather than treated as independent observations.
- **Statistical significance is not practical significance:** no practical-effect threshold was pre-registered.

## 6. Summary of research findings

| RQ | Traceable finding |
|---|---|
| 1 | HNSW p50 is 4.11–8.01× faster at 10k and 8.17–39.63× faster at 50k for valid configurations. |
| 2 | Recall@5 loss is −0.00007 to −0.0051 at 10k and −0.0004 to −0.0171 at 50k. |
| 3 | Within-size recall/accuracy correlation is positive; classification differences are not significant after Holm. |
| 4 | Higher ef raises recall and latency, with diminishing returns after ef=50/100. |
| 5 | Exact latency grows ~4.7× from 10k to 50k; HNSW latency remains nearly flat. |
| 6 | Indexes are 4.984 MB (10k) and 24.875 MB (50k); build costs are 1.48 s and 3.50 s. |
| 7 | Natural planner rejects 10k/ef=400; controlled mode obtains an index scan but remains separate. |
| 8 | Exact is preferable for deterministic recall and small pools. |
| 9 | HNSW is preferable for larger pools when small measured recall losses are acceptable. |

All claims are traceable to `docs/results/validated/summary.csv`, `statistical_summary.json` or the validated Phase 3 reports. No raw measurement was modified. Analysis stops here.
