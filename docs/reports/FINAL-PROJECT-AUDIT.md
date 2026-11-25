# Final Project Audit

**Audited base commit:** `49a5abab0507aa06d472f1d64dcd787c4e65e07c`  
**Branch:** `phase3-benchmark-design`  
**Audit date:** 25 November 2026  
**Final decision:** **NOT READY FOR SUBMISSION** — one unresolved security issue remains

---

## 1. Concise final summary

### 1.1 Research objectives

The project investigated the trade-off between exact KNN and approximate HNSW KNN for supervised classification using PostgreSQL and pgvector. It targeted retrieval accuracy, classification performance, query latency, index construction, storage, training-pool size and HNSW search effort.

### 1.2 Implemented methodology

A leakage-free Covertype pipeline created immutable 10k/50k derived pools with recorded preprocessing state. Exact search used the pgvector distance function; HNSW used the index-compatible operator with catalog, ef and per-query plan verification. Three balanced rounds × three repetitions were run over 3,000 held-out query ids. Natural planner and controlled HNSW modes were kept separate. A pre-registered statistical protocol used query-level paired comparisons.

### 1.3 Main verified findings

- HNSW p50 speed-ups were 4.11–8.01× at 10k and 8.17–39.63× at 50k for valid configurations.
- Recall@5 losses were small at higher ef and largest at ef=10 (−0.0171 on 50k).
- Classification differences were not significant after Holm correction.
- Natural 10k/ef=400 used sequential scans and was excluded as a failed HNSW access-path check.
- Index size was 4.984 MB / 24.875 MB; fresh build times were 1,478.5 ms / 3,500.2 ms.

### 1.4 Scientific contributions

The project contributes a database-native exact/ANN comparison tied to classification outcomes, strict planner/access-path verification, complete provenance, paired statistical analysis, and fully reproducible validated artifacts.

### 1.5 Limitations

Single dataset family, L2, k=5, warm cache, client-side timing, three rounds, one failed natural cell, no per-class exports, no cold-cache/concurrency study. No universal superiority claim is supported.

### 1.6 Test results

Full backend suite on freshly initialized isolated databases: **272 passed, 1 skipped, 1 warning**.

### 1.7 Reproducibility status

Reproducibility instructions are complete in `docs/reports/REPRODUCIBILITY-GUIDE.md`. Validated outputs and figures/tables regenerated deterministically.

### 1.8 Final Git commit

The audited base commit is `49a5aba`. The commit containing this audit and the two companion documents is reported in the final response after it is created.

### 1.9 Deliverable locations

- Thesis: `docs/thesis/THESIS.docx`, `docs/thesis/THESIS.md`
- Bibliography: `docs/thesis/references.bib`
- Figures: `docs/thesis/figures/`, `docs/results/figures/`
- Tables: `docs/thesis/tables/`, `docs/results/tables/`
- Validated results: `docs/results/validated/`
- Reports: `docs/reports/`
- Reproducibility: `docs/reports/REPRODUCIBILITY-GUIDE.md`
- Audit documents: this file, `FINAL-DELIVERY-CHECKLIST.md`

---

## 2. Repository integrity

### 2.1 Git history and state

| Check | Result |
|---|---|
| Branch | `phase3-benchmark-design` |
| Audited HEAD | `49a5aba` |
| Recent history | Phase 2 remediation → Phase 3 protocol → diagnostic → benchmark → validation → analysis → thesis |
| Stash | `phase3-preflight-unrelated` still present and untouched |
| Working tree | Only internal memory notes were untracked at audit start; README and three audit documents were updated during the audit |
| Remote | `origin` configured; no push performed |

History was not reset, rebased or rewritten.

### 2.2 Credentials

No live credential was found in tracked current files. `backend/.env.example` contains the placeholder `change-me-before-running`; the benchmark protocol derives `PGPASSWORD` from the local `.env` via a shell expression and does not print it.

**Unresolved issue:** `README.md` states that a password from an earlier project version remains in Git history. This is an accidental credential. It must be rotated in PostgreSQL. Historical commits were not rewritten. Because the criterion says no accidental credentials, this prevents a READY decision even though no current secret was found in the working tree.

### 2.3 Generated files

- No tracked `.pyc`, `.log` or `__pycache__` files.
- Untracked memory notes are internal project data, not generated build output.
- Historical figures `figure_1_recall.png` … `figure_6_classes.png` remain tracked and are used by legacy documents; they are not used by the current thesis.
- No unnecessary generated files were added by the audit; validators and figure generation were redirected to OS temp directories.

### 2.4 Source files

All required source files are tracked: backend app, experiment runner, validators, figure generator, thesis builder, thesis source, bibliography, validated artifacts and final DOCX.

### 2.5 Phase 1 guarantees

Phase 1 tests for transactional variant creation, rollback, preprocessing state, training subsampling, API contracts and configuration security passed in the full suite.

### 2.6 Database schema consistency

Live database: `knn_knn_platform`, PostgreSQL 18.0, pgvector 0.8.7. Core tables and variant tables are present. `experiments` contains `status`, `retrieval_method`, `used_index_scan` and `error_message`; `planner_mode` is stored in JSONB `config`/`environment`, not as a dedicated column. 38 Phase 3 experiments are present, matching 30 natural + 8 controlled runs.

---

## 3. Scientific reproducibility

The ten reproducibility steps in the prompt are documented with exact commands in `REPRODUCIBILITY-GUIDE.md`: software, dependencies, PostgreSQL/pgvector, local environment variables, dataset ingestion, preprocessing, derived datasets, index creation, approved experiments, raw artifacts, metric recalculation, figures/tables and thesis build.

Verification results:

- Phase 3 validator re-run into a temp directory: `PASS_WITH_DOCUMENTED_EXCLUSIONS`, 35 validated summary rows, zero errors.
- Re-run `summary.csv` byte-identical to committed output; `statistical_summary.json` identical except generation timestamp.
- Eight new figures and six tables regenerated byte-identically.
- DOCX rebuilt into a temp path with identical structure: 309 paragraphs, 6 tables, 8 images.

Caveat: a fresh database must be initialized before tests; reusing a dirty test database produced a duplicate dataset-id error, which disappeared on freshly initialized databases.

---

## 4. Thesis verification

- Numerical claims were checked against `docs/results/validated/` and Phase 4 tables; key values matched.
- Figures 1–8 and Tables 1–3 are numbered, captioned and referenced.
- Equations (1)–(7) are present.
- Citations [1]–[11] and 11 BibTeX entries are present.
- DOIs were verified against Crossref, JMLR and UCI. Corrections from the original proposal were applied.
- The active DOCX was inspected through the local editor service: 8 embedded images were reported. The service's outline query returned an empty list despite python-docx and DOCX XML confirming nine Heading 1 entries; this appears to be an editor-instance cache discrepancy, not a document defect.
- Conclusions do not claim universal superiority and retain the failed natural 10k/ef=400 cell as an exclusion.
- Per-class and confusion-matrix claims are explicitly unsupported because those matrices were not exported to validated outputs.

---

## 5. Automated verification

| Check | Result |
|---|---|
| Full backend pytest on fresh isolated DBs | 272 passed, 1 skipped, 1 warning |
| Initial reused-DB pytest | 271 passed, 1 skipped, 1 duplicate-id setup error — environment residue, resolved on fresh DB |
| `compileall` for backend, scripts, thesis and tools | passed |
| Ruff | not installed; not run |
| Mypy | not installed; not run |
| Flake8 / Pylint | not installed; not run |
| Phase 3 custom validator | PASS, zero errors |
| Legacy `verify_results.py` | 18 expected failures (old six-ef matrix + rejection of failed HNSW rows); output preserved |
| Figure/table regeneration | 8/8 figures and 6/6 tables byte-identical |
| Thesis DOCX rebuild | structure identical |
| Local editor DOCX image check | 8 embedded images |

The unverified lint/type tools are recorded as unavailable, not as passes.

---

## 6. Required documentation inventory

| Requirement | Location | Status |
|---|---|---|
| README | `README.md` | present; updated during audit |
| Environment/setup | `README.md`, reproducibility guide | present |
| Database setup | `README.md`, reproducibility guide | present |
| Dataset preparation | `README.md`, reproducibility guide | present |
| Benchmark execution | reproducibility guide, protocol | present |
| Known limitations | protocol, validation report, thesis, audit | present |
| Experiment configuration | protocol, full benchmark report, thesis appendices | present |
| Research methodology | protocol, thesis chapters 3–4 | present |
| Results and figures | `docs/results/`, thesis folders | present |
| Thesis source | `docs/thesis/THESIS.md`, `references.bib` | present |
| Final thesis document | `docs/thesis/THESIS.docx` | present |
| Reproducibility guide | `docs/reports/REPRODUCIBILITY-GUIDE.md` | present |
| Final audit/checklist | `docs/reports/` | present |

---

## 7. Acceptance criteria

| Criterion | Verdict |
|---|---|
| Implementation verified | PASS |
| Required experiments complete/accounted for | PASS |
| Results independently validated | PASS |
| Figures and tables reproducible | PASS |
| Thesis claims match validated results | PASS |
| Citations verified | PASS |
| Required tests pass | PASS on fresh DB |
| Reproducibility instructions complete | PASS |
| No accidental credentials | **FAIL** — historical password remains in Git history |
| No unnecessary generated files | PASS |
| Missing source files | PASS |

## 8. Final decision

**NOT READY FOR SUBMISSION**

All scientific and implementation criteria pass, but the historical credential in Git history violates the explicit "no accidental credentials" criterion. The credential must be rotated in PostgreSQL; repository history should not be rewritten without a separate explicit decision. After rotation, the project may be re-audited and marked ready without changing any scientific result.
