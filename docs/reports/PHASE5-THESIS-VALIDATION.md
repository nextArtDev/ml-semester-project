# Phase 5 — Thesis Validation Report

**Starting commit:** `deb775851ace042ecd8cf294840417e40f54f578`  
**Branch:** `phase3-benchmark-design`  
**Thesis source:** `docs/thesis/THESIS.md`  
**DOCX output:** `docs/thesis/THESIS.docx`  
**Bibliography:** `docs/thesis/references.bib`  
**Validation status:** PASS

---

## 1. Source-of-truth audit

The thesis was written from the approved proposal (`docs/ONE-PAGE-PROPOSAL.md`, read from the preserved stash without modifying it), `PHASE3-BENCHMARK-PROTOCOL.md`, `PHASE3-RESULTS-VALIDATION.md`, `PHASE4-SCIENTIFIC-ANALYSIS.md`, the verified implementation/tests, and validated figures/tables. Smoke-test metrics were explicitly excluded and checked for absence.

The repository's older `docs/tools/build_final_thesis.py` was inspected but **not used**: it reads the historical `docs/results/sweep.csv`, assumes the old six-ef matrix and regenerates the prior figures. Using it would violate the validated-source requirement. A new `docs/thesis/build_thesis_docx.py` uses the repository's existing `docx_kit` styling helpers and reads only `THESIS.md` plus the copied validated figures.

## 2. Inputs and outputs

Copied into the thesis folder:

- 8 figures: `docs/thesis/figures/figure_01_...png` through `figure_08_...png`
- 6 validated tables: `docs/thesis/tables/*.csv`
- thesis source, bibliography and DOCX under `docs/thesis/`

The validated raw artifacts in `docs/results/validated/` and `docs/results/tables/` were not modified.

## 3. Structural checks

| Check | Result |
|---|---|
| Chapters 1–7 present | passed |
| References section present | passed |
| Appendices present | passed |
| Table of contents field in DOCX | passed |
| Chapter heading levels in DOCX | passed: seven chapters, References, Appendices |
| Page-number field in DOCX footer | passed |
| Figure captions Figure 1–Figure 8 | passed and sequential |
| Figure files resolve | 8/8 passed |
| DOCX embedded images | 8 |
| Table captions Table 1–Table 3 | passed and sequential |
| DOCX tables | 6 (captioned result tables plus appendix/configuration tables) |
| Equations tagged (1)–(7) | passed |
| Citations [1]–[11] used | passed |
| BibTeX entries | 11 |
| No smoke-test metrics in thesis | passed |
| Validated key numbers present | passed |

The DOCX XML was inspected: `a:graphicData` elements all carry the required `uri` attribute, so images should open correctly in Word.

## 4. Reference verification

Bibliographic details and DOIs were checked against Crossref, JMLR, UCI and official project documentation:

- Cover & Hart (1967): `10.1109/TIT.1967.1053964` — verified.
- Indyk & Motwani (1998): `10.1145/276698.276876` — verified; the frequently mistaken `276781` is a different paper.
- Malkov *et al.* (2014): `10.1016/j.is.2013.10.006`, pp. 61–68 — verified.
- Malkov & Yashunin (2020): `10.1109/TPAMI.2018.2889473`, 42(4), 824–836 — verified.
- Blackard & Dean (1999): `10.1016/S0168-1699(99)00046-0` — verified.
- UCI Covertype: DOI `10.24432/C50K5N`, 581,012 instances, 54 features, seven classes — verified on the official UCI page.
- PostgreSQL 18 documentation and pgvector README — verified as official sources.
- Pedregosa *et al.* (2011) — verified on JMLR.
- Johnson *et al.* FAISS paper — verified as *IEEE Transactions on Big Data*, 7(3), 535–547, 2021, DOI `10.1109/TBDATA.2019.2921572`.
- Aumüller *et al.* ANN-Benchmarks — verified as *Information Systems*, 87, 101374, 2020, DOI `10.1016/j.is.2019.02.006`.

No university template or formatting standard was claimed.

## 5. Results agreement

The main results table was checked against validated values. Verified examples include: 10k exact p50 2.386 ms; 10k HNSW ef=10 p50 0.298 ms; 50k exact p50 11.278 ms; 50k HNSW ef=50 p50 0.416 ms and Recall@5 0.9975; index sizes 4.984 MB and 24.875 MB; build times 1,478.5 ms and 3,500.2 ms. All are traceable to `docs/results/validated/summary.csv` and Phase 4 tables.

The thesis preserves the exclusion of the three natural 10k/ef=400 sequential-scan failures and keeps controlled HNSW results separate.

## 6. Limitations of the validation

- Figure numbering, table numbering, citation presence and result-key agreement were checked programmatically; semantic grammar was reviewed manually but was not machine-graded.
- Word's TOC is a field and must be refreshed in Word to populate page numbers.
- Per-class and confusion-matrix details are unavailable in validated outputs; the thesis explicitly states this and makes no per-class claim.
- The DOCX was generated from the preserved Markdown source rather than the older thesis builder because that builder's inputs are historical and incompatible with the approved matrix.

## 7. Final decision

The thesis draft is internally consistent, traceable to validated results, and academically complete within the stated scope. The generated DOCX is editable and structurally valid. No further scientific or formatting work was performed after this validation.
