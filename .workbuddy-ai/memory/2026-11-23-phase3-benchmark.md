# 2026-11-23 — Phase 3 full benchmark execution note

- Prompt 5 executed with option A on branch `phase3-benchmark-design`. Commit `e8cdb87` contains `PHASE3-FULL-BENCHMARK-REPORT.md` and all raw artifacts in `docs/results/phase3/`.
- Unique run IDs: `phase3-natural-v1-20261123T224455` and `phase3-controlled-diagnostic-v1-20261123T224455`.
- 30 natural + 8 controlled experiments; 342,000 latency rows. Three natural 10k/ef=400 HNSW runs failed the access-path check as specified by option A and are excluded from valid HNSW summaries. All controlled runs used the index.
- First runner invocation and one resume invocation hit the 10-minute command limit; only missing configurations were resumed, no completed experiments were deleted or overwritten. Artifacts were rebuilt from preserved DB rows.
- Standard `verify_results.py` fails because it hard-codes the old six-ef matrix and rejects failed HNSW rows. A custom option-A validator passed with zero errors; results retained in `docs/results/phase3/validation_summary.json` and `standard_verifier_output.json`.
- `phase3-preflight-unrelated` stash was left untouched; memory note written to this separate dated file to avoid modifying the stashed memory file.

## Prompt 6 validation (same day)

- Executed independent validation with `backend/scripts/validate_phase3_results.py`; committed as `a31f972` together with `docs/reports/PHASE3-RESULTS-VALIDATION.md`, `docs/results/validated/summary.csv`, and `statistical_summary.json`.
- Starting commit `e8cdb87`; validation status PASS_WITH_DOCUMENTED_EXCLUSIONS. 35 validated summary rows (27 natural + 8 controlled); 3 natural 10k/ef=400 failures excluded. Recomputation discrepancies ≤ 3.33e-16; full backend tests 272 passed, 1 skipped on isolated DB.
- Statistical audit: 7 valid paired natural comparisons; Wilcoxon/Holm latency p-values underflow to <1e-300, rank-biserial −1 for all; recall gaps negative; McNemar classification differences not significant after Holm. GO for Prompt 7 with documented exclusions. Prompt 7 not started.

## Prompt 7 scientific analysis (same day)

- Generated `docs/reports/PHASE4-SCIENTIFIC-ANALYSIS.md`, 8 publication figures in `docs/results/figures/`, 6 tables in `docs/results/tables/`, and reproducible `scripts/generate_research_figures.py`; committed as `deb7758`.
- Used only validated outputs. Natural and controlled modes kept separate. Per-class/confusion-matrix outputs were not in validated outputs, so `per_class_results.csv` explicitly documents non-availability instead of fabricating results.
- Findings: HNSW p50 speed-ups 4.11–8.01× at 10k and 8.17–39.63× at 50k; recall losses −0.00007 to −0.0051 (10k) and −0.0004 to −0.0171 (50k); classification differences not significant after Holm; index sizes 4.984/24.875 MB and build times 1.48/3.50 s. Stopped after analysis.
