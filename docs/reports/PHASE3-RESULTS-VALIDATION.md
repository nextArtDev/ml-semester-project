# Phase 3 — Results Validation and Statistical Audit

**Branch:** `phase3-benchmark-design`  
**Starting commit:** `e8cdb87fd58445d7089a2a2e5071b41fc3cb4330`  
**Natural run:** `phase3-natural-v1-20261123T224455`  
**Controlled run:** `phase3-controlled-diagnostic-v1-20261123T224455`  
**Validation status:** PASS with documented exclusions  
**Decision for Prompt 7:** **GO**, conditional on the exclusions below

---

## 1. Scope and input files

Starting state was independently verified: branch `phase3-benchmark-design`, HEAD `e8cdb87`, all raw Phase 3 artifacts readable, and both target outputs absent. The working tree contained one untracked internal memory note (`.workbuddy-ai/memory/2026-11-23-phase3-benchmark.md`) and the pre-existing `phase3-preflight-unrelated` stash; neither was modified.

Files inspected:

- `docs/reports/PHASE3-BENCHMARK-PROTOCOL.md`
- `docs/reports/PHASE3-FULL-BENCHMARK-REPORT.md`
- `docs/results/phase3/sweep.csv`, `sweep.json`
- `docs/results/phase3/repetition_details.csv`, `repetition_summary.csv`
- `docs/results/phase3/query_samples.json`, `benchmark_orders.json`, `variant_ids.json`
- `docs/results/phase3/controlled_diagnostic.json`
- `docs/results/phase3/validation_summary.json`, `standard_verifier_output.json`
- PostgreSQL records in `knn_knn_platform` for both Phase 3 run IDs

No raw Phase 3 artifact was modified.

Validation code: `backend/scripts/validate_phase3_results.py`.

## 2. Validation methodology

The validator independently:

1. loaded all raw artifacts and reconciled CSV/JSON records;
2. queried the production database read-only for the 38 Phase 3 experiments, 342,000 latency rows and 114 repetition rows;
3. recovered selected training ids from `fv_116` joined to each variant by `row_index` and recomputed their SHA-256;
4. verified query ids, class labels and neighbour labels from the variant tables;
5. recomputed access-path classes from persisted plans, flags and status;
6. recomputed latency statistics, Recall@5, classification metrics, confusion matrices, per-class metrics and repetition summaries directly from `query_latencies`;
7. performed the protocol's paired statistical analysis with query id as the unit of inference.

Documented tolerances: absolute difference ≤ `1e-6` for all numeric recomputation comparisons.

## 3. Artifact and database reconciliation

| Check | Result |
|---|---|
| Artifact files present/readable | 10/10 |
| Sweep rows | 30 CSV = 30 JSON, identical field sets and values |
| Repetition detail rows | 90 (30 experiments × 3 repetitions) |
| Repetition summary rows | 300 (30 × 10 metrics) |
| Query sample entries | 30 natural experiments × 3,000 ids |
| Benchmark orders | 6 size/round entries, each a 5-configuration permutation with 5 experiment ids |
| Variant records | 2 variants, 10,000 and 50,000 selected ids; hashes recomputed successfully |
| Controlled diagnostic records | 8 |
| Database experiments | 38 (30 natural + 8 controlled) |
| Latency rows | 342,000 (38 × 9,000); 342 distinct experiment references |
| Repetition rows | 114 (38 × 3) |
| Orphan latency/repetition rows | 0 |
| Duplicate experiment keys | 0 |
| Parent fingerprint | `6baa8fa283ea5a8267d3e964eedf5d2a`, unchanged |
| Historical experiments 382–384 | unchanged |
| HNSW index catalog | `hnsw` / `vector_l2_ops` on both variants; stored sizes match `pg_relation_size` |

`scripts/verify_results.py` still reports 18 checks as failures; its output is preserved in `standard_verifier_output.json`. Both causes are known incompatibilities, not data defects: it hard-codes the older six-ef matrix and rejects any failed HNSW row. The verifier was not modified and no artifact was adjusted to satisfy it.

## 4. Access-path validation

| Class | Count | Treatment |
|---|---:|---|
| Valid exact sequential scans | 6 | included as baselines |
| Valid natural HNSW index scans | 21 | included in natural HNSW comparisons |
| Natural sequential-scan HNSW failures | 3 | retained as diagnostics; excluded from HNSW performance |
| Valid controlled HNSW index scans | 8 | included only as separate controlled diagnostics |
| Incomplete/unverified | 0 | — |

The three excluded experiments are **388, 392 and 396**: 10,000-row pool, ef=400, natural planner, rounds 1–3. Each reports `used_index_scan=false`, `used_sequential_scan=true` and an explicit error message. No controlled result was pooled with natural results.

## 5. Metric recomputation

Recomputed values were compared against stored experiment, repetition and artifact values.

| Metric family | Maximum absolute discrepancy |
|---|---:|
| p50 / p95 / p99 / min / max / avg latency | 0.0 |
| Recall@5 | 3.33 × 10⁻¹⁶ |
| Accuracy / macro precision / macro recall / macro F1 | 0.0 |
| Per-repetition metrics | 0.0 |
| Confusion matrices and per-class metrics | exact match |
| Repetition-summary mean/median/sd/min/max/CI | 0.0 |

No discrepancy exceeded `1e-6`; none required replacing a stored value. Neighbour labels were joined from `fv_<variant>` by retrieved neighbour id, and query labels from `tv_<variant>` by the persisted `evaluated_query_ids`; label mappings matched the parent dataset's class labels.

## 6. Statistical audit

### Design

- Unit of inference: query id within pool size.
- Repeated measurements: 3 rounds × 3 repetitions = 9 passes per query.
- Aggregation: per-query median latency and per-query majority classification decision across the 9 passes.
- Paired comparisons: each HNSW configuration against the exact baseline using the same 3,000 query ids.
- Bootstrap: 10,000 resamples of query ids, seed `20261123`.
- Multiple comparisons: Holm correction across ef levels within each size, separately for latency and classification families.

### Results for valid natural comparisons

Median latency differences (`HNSW − Exact`) were negative for all seven valid comparisons:

| Size | ef | Median Δ latency (ms) | 95% bootstrap CI | Holm p | Rank-biserial | Mean Δ Recall@5 (95% CI) | Δ accuracy | McNemar Holm p |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 10,000 | 10 | −2.043 | [−2.050, −2.030] | < 1e-300 | −1.000 | −0.0051 [−0.0070, −0.0034] | −0.0010 | 1.000 |
| 10,000 | 50 | −1.935 | [−1.944, −1.926] | < 1e-300 | −1.000 | −0.0002 [−0.00047, 0.0000] | +0.0003 | 1.000 |
| 10,000 | 100 | −1.731 | [−1.740, −1.720] | < 1e-300 | −1.000 | −0.00007 [−0.0002, 0.0000] | 0.0000 | 1.000 |
| 50,000 | 10 | −10.927 | [−10.932, −10.923] | < 1e-300 | −1.000 | −0.0171 [−0.0215, −0.0128] | −0.0020 | 1.000 |
| 50,000 | 50 | −10.803 | [−10.808, −10.797] | < 1e-300 | −1.000 | −0.0025 [−0.0043, −0.0011] | −0.0007 | 1.000 |
| 50,000 | 100 | −10.654 | [−10.660, −10.647] | < 1e-300 | −1.000 | −0.0012 [−0.0022, −0.0004] | −0.0003 | 1.000 |
| 50,000 | 400 | −9.798 | [−9.808, −9.789] | < 1e-300 | −1.000 | −0.0004 [−0.0008, −0.00007] | 0.0000 | 1.000 |

Wilcoxon p-values underflowed to `0.0` in double precision and are reported as `< 1e-300`, not as literal zero. Rank-biserial `−1` means every paired query was faster under HNSW in the aggregated comparison.

Recall differences are negative: exact retrieval is the reference and HNSW sacrifices some neighbour agreement, most visibly at ef=10 and larger pools. Classification accuracy differences are small and McNemar tests are not significant after Holm correction. These are paired observations on one dataset and do not establish universal superiority.

The approved protocol did not pre-register a practical-significance threshold, so this audit reports effect magnitudes and intervals without declaring practical superiority.

### Controlled HNSW diagnostic

All eight controlled runs are valid index scans and completed, but they are descriptive only. They were not paired against exact latencies or pooled with natural results.

## 7. Included and excluded measurements

Included in `docs/results/validated/summary.csv`: 35 records — 6 exact baselines, 21 valid natural HNSW runs and 8 controlled HNSW diagnostics.

Excluded from valid HNSW performance comparisons: experiments 388, 392, 396 (10,000/ef=400 natural). They remain visible in raw artifacts and the database with full diagnostics. Their latency must not be interpreted as HNSW latency.

## 8. Discrepancies and limitations

- No recomputation discrepancy exceeded the tolerance.
- The standard verifier's 18 failures are explained by its outdated ef matrix and strict HNSW status rule; it was left unchanged.
- `query_samples.json` is a natural-runner artifact only; controlled query ids were verified directly from database records.
- Results come from one machine, one dataset family, L2, k=5 and a warm buffer cache. Cold-cache behavior is not measured.
- Latency is client-side `execute + fetch` time including round trip; it is not pure server execution time.
- Three rounds provide limited run-to-run replication; query-level bootstrap estimates describe this query set, not a population of all possible queries.
- Repeated passes are correlated; they were aggregated per query rather than treated as independent observations.
- The 10k/ef=400 natural cell remains a confirmed planner fallback; no result was substituted for it.
- Statistical significance of latency differences is not evidence of practical superiority without a pre-registered threshold.

## 9. Final decision

**GO for Prompt 7** using only `docs/results/validated/summary.csv` and `docs/results/validated/statistical_summary.json`, with the three documented exclusions and the separation of natural and controlled modes maintained.

**NO-GO** for interpreting the failed natural 10k/ef=400 runs as HNSW measurements or for pooling controlled-HNSW results into natural comparisons.

Tests executed: full backend suite on an isolated test database — **272 passed, 1 skipped** (pre-existing cosine-collinearity skip), 1 deprecation warning. Validation script executed successfully with zero validation errors.

Prompt 7 was not started.
