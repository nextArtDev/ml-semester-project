# Phase 3 — Isolated Diagnostic Re-test Report

**Branch:** `phase3-benchmark-design` @ `6895f24`  
**Purpose:** resolve the Phase 2 NO-GO precondition before the full benchmark  
**Status:** diagnostic complete — **full benchmark not executed**  
**Next action:** review and explicit approval required

---

## 1. Preflight state

| Check | Result |
|---|---|
| Branch / commit | `phase3-benchmark-design` @ `6895f24` (approved protocol commit) |
| Working tree | Stashed as `stash@{0}: phase3-preflight-unrelated`; tree was clean before the diagnostic |
| Phase 2 GO report | GO for remediation; full benchmark NO-GO pending this diagnostic |
| Approved protocol | `docs/reports/PHASE3-BENCHMARK-PROTOCOL.md`, committed at `6895f24` |
| PostgreSQL / pgvector | 18.0 / 0.8.7 |
| Disk / memory | 41 GB free on C:; sufficient for the protocol estimate |
| Backup | Not created; no production data was modified |
| Production data | Dataset 116, experiments and historical records untouched |

The three unrelated working-tree files (two memory logs and `docs/ONE-PAGE-PROPOSAL.md`) were stashed and remain stashed. They will be restored only after the final full-benchmark report, as agreed.

## 2. Isolated diagnostic setup

A dedicated database `test_phase3_diag` was created. Its schema was built from `app.db.schema`, and dataset 116 was copied **read-only from production** into the isolated database:

- 50,000 training rows, 3,000 held-out rows, 54 dimensions.
- Derived variants created with `build_preprocessed_variant(116, size, seed=0)` for 10,000, 20,000 and 50,000 rows.
- HNSW index built with the protocol's fixed `m=16`, `ef_construction=64`.
- One query per variant was used for plan verification; no full benchmark pass was run.
- Natural and controlled runs were separate and recorded with distinct `planner_mode` values.

## 3. Results

| Pool size | Variant | ef=10 | ef=50 | ef=100 | ef=400 |
|---:|---:|---|---|---|---|
| 10,000 | 2 | HNSW index scan (`completed`) | HNSW index scan (`completed`) | HNSW index scan (`completed`) | **Seq Scan (`failed`)** |
| 20,000 | 3 | HNSW index scan (`completed`) | HNSW index scan (`completed`) | HNSW index scan (`completed`) | HNSW index scan (`completed`) |
| 50,000 | 4 | HNSW index scan (`completed`) | HNSW index scan (`completed`) | HNSW index scan (`completed`) | HNSW index scan (`completed`) |

Estimated Limit costs for the natural plans:

| Pool size | ef=10 | ef=50 | ef=100 | ef=400 |
|---:|---:|---:|---:|---:|
| 10,000 | 59.03..61.07 | 168.69..170.68 | 286.19..288.12 | 660.10..660.11 (Seq Scan) |
| 20,000 | 62.57..64.67 | 180.58..182.65 | 307.01..309.05 | 956.74..958.62 |
| 50,000 | 67.16..69.22 | 196.03..198.07 | 334.10..336.13 | 1043.62..1045.57 |

### Controlled HNSW mode

All 12 controlled diagnostic runs (3 sizes × 4 ef values) used the exact HNSW index, persisted `planner_mode='controlled_hnsw'`, `applied_ef_search` equal to the requested value, and `status='completed'`. None was mixed with a natural-mode result.

### Provenance

| Size | Variant | Dataset fingerprint | Selected IDs | Selected-ID SHA-256 (first 16) | Preprocessing-state SHA-256 (first 16) |
|---:|---:|---|---:|---|---|
| 10,000 | 2 | `f13de289c970d19e1c00c95afb49461e` | 10,000 | `6205c393930ed035…` | `c51634ae1fffc922…` |
| 20,000 | 3 | `bfe85f30e626ddac6df79d821835d8f9` | 20,000 | `4b89855f71939cde…` | `71bdd17d9d13e09b…` |
| 50,000 | 4 | `4ab59c99ecc396826c511e3d5ea06d17` | 50,000 | `b8b8dd49035f090f…` | `24b7e4ec3f6537b2…` |

Full hashes and per-run experiment ids are retained in `docs/results/phase3-diagnostic/diagnostic.json`.

## 4. Interpretation

The Phase 2 diagnosis is confirmed at the exact pool sizes in the approved protocol:

1. Natural planner behavior is valid for ef=10/50/100 at 10,000, 20,000 and 50,000 rows.
2. At **10,000 rows and ef=400**, pgvector's ef-dependent HNSW cost exceeds the sequential-scan alternative, so the natural planner selects a sequential scan. The Phase 2 remediation correctly records this run as `failed`; it is not an HNSW measurement.
3. Controlled HNSW mode obtains a verified index scan for that cell, but it is a separate result category and must not replace the natural result.
4. No evidence was found that any other approved cell will fail for planner reasons. Runtime failures of individual configurations must still be handled as specified in Prompt 5.

## 5. Impact on the approved protocol

The approved natural matrix includes 10,000 × ef=400. Under the confirmed planner behavior, that cell is expected to be a failed/non-measurement. **No experimental design change has been made.** Three options require explicit approval before the full benchmark:

| Option | Consequence |
|---|---|
| A — Execute exactly as approved | The 10k/ef=400 natural run will be reported as failed and excluded from valid HNSW summaries; all other cells proceed unchanged. |
| B — Amend protocol | Keep 10k/ef=400 only in the controlled diagnostic category; do not count it as a natural HNSW cell. This changes the approved design and requires explicit approval. |
| C — Stop | No full benchmark until a different design is approved. |

## 6. Conclusion

The Phase 2 NO-GO diagnostic precondition has been executed and documented. The full benchmark remains **not started**. Approval is required both for the full benchmark and for the handling of the 10k/ef=400 natural cell before execution continues.
