"""SQL schema for the KNN classification platform.

Design note - why one table per dataset?

pgvector's HNSW index requires the indexed column to have a **fixed
dimension** (``vector(54)`` rather than a bare ``vector``). It also applies
any residual filter *after* the index scan, so a single shared table holding
several datasets would retrieve neighbours from the wrong dataset and
silently corrupt recall@K.

The platform therefore creates a dedicated pair of tables per dataset:

* ``fv_<dataset_id>`` - the training pool (the HNSW index lives here).
* ``tv_<dataset_id>`` - the held-out query pool.

Both use a dimensionally-typed ``vector(dim)`` column, which lets PostgreSQL
choose a clean HNSW index scan with no residual filter.
"""

from __future__ import annotations

SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------------------
-- Datasets
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS datasets (
    id              BIGSERIAL PRIMARY KEY,
    name            TEXT        NOT NULL,
    source          TEXT        NOT NULL DEFAULT 'upload',
    original_name   TEXT,
    n_rows          INTEGER     NOT NULL DEFAULT 0,
    n_features      INTEGER     NOT NULL DEFAULT 0,
    n_classes       INTEGER     NOT NULL DEFAULT 0,
    dimension       INTEGER     NOT NULL DEFAULT 0,
    class_labels    JSONB       NOT NULL DEFAULT '[]'::jsonb,
    preprocessing   JSONB       NOT NULL DEFAULT '{}'::jsonb,
    is_seeded       BOOLEAN     NOT NULL DEFAULT FALSE,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS datasets_name_idx ON datasets (name);

-- ---------------------------------------------------------------------------
-- Experiments
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS experiments (
    id                  BIGSERIAL PRIMARY KEY,
    dataset_id          BIGINT      NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
    name                TEXT,
    retrieval_method    TEXT        NOT NULL CHECK (retrieval_method IN ('exact', 'hnsw')),
    k                   INTEGER     NOT NULL CHECK (k > 0),
    distance_metric     TEXT        NOT NULL CHECK (distance_metric IN ('l2', 'cosine', 'l1')),
    hnsw_ef_search      INTEGER,
    hnsw_m              INTEGER,
    hnsw_lists          INTEGER,
    train_size          INTEGER     NOT NULL DEFAULT 0,
    test_size           INTEGER     NOT NULL DEFAULT 0,
    index_build_ms      DOUBLE PRECISION,
    recall_at_k         DOUBLE PRECISION,
    accuracy            DOUBLE PRECISION,
    macro_f1            DOUBLE PRECISION,
    precision_macro     DOUBLE PRECISION,
    recall_macro        DOUBLE PRECISION,
    latency_avg_ms      DOUBLE PRECISION,
    latency_p50_ms      DOUBLE PRECISION,
    latency_p95_ms      DOUBLE PRECISION,
    latency_p99_ms      DOUBLE PRECISION,
    total_runtime_s     DOUBLE PRECISION,
    used_index_scan     BOOLEAN,
    confusion_matrix    JSONB       NOT NULL DEFAULT '{}'::jsonb,
    per_class_metrics   JSONB       NOT NULL DEFAULT '[]'::jsonb,
    config              JSONB       NOT NULL DEFAULT '{}'::jsonb,
    status              TEXT        NOT NULL DEFAULT 'pending'
                              CHECK (status IN ('pending', 'running', 'completed', 'failed')),
    error_message       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at        TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS experiments_dataset_idx
    ON experiments (dataset_id, created_at DESC);

-- ---------------------------------------------------------------------------
-- Per-query latency samples (for latency distribution charts)
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS query_latencies (
    id            BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT      NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    query_index   INTEGER     NOT NULL,
    latency_ms    DOUBLE PRECISION NOT NULL,
    retrieved_ids JSONB       NOT NULL DEFAULT '[]'::jsonb
);

CREATE INDEX IF NOT EXISTS query_latencies_experiment_idx
    ON query_latencies (experiment_id);
"""


def migration_statements() -> list[str]:
    """Split the schema blob into individual statements."""
    statements: list[str] = []
    for raw in SCHEMA_SQL.split(";"):
        stmt = "\n".join(
            line for line in raw.strip().splitlines() if not line.strip().startswith("--")
        ).strip()
        if stmt:
            statements.append(stmt)
    return statements
