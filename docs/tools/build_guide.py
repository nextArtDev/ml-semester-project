"""Generate the project guide Word document.

Run from the repository root::

    python docs/tools/build_guide.py

The guide walks through every layer of the platform — database, backend and
frontend — explaining each module, each function and each design decision.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.shared import Pt  # noqa: E402

from docx_kit import (  # noqa: E402
    add_page_number_footer,
    add_table_of_contents,
    bullet,
    bullet_list,
    caption,
    code,
    h1,
    h2,
    h3,
    h4,
    new_document,
    note,
    numbered,
    p,
    page_break,
    table,
)

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "Project-Guide.docx"


# ---------------------------------------------------------------------------
# Part 0
# ---------------------------------------------------------------------------
def build_cover(doc) -> None:
    for _ in range(4):
        doc.add_paragraph()

    paragraph = doc.add_paragraph()
    paragraph.alignment = 1
    run = paragraph.add_run("PROJECT GUIDE")
    run.bold = True
    run.font.size = Pt(30)

    paragraph = doc.add_paragraph()
    paragraph.alignment = 1
    run = paragraph.add_run(
        "PostgreSQL-Based KNN Classification Platform"
    )
    run.bold = True
    run.font.size = Pt(17)

    paragraph = doc.add_paragraph()
    paragraph.alignment = 1
    run = paragraph.add_run(
        "A function-by-function tour of the database, the backend and the frontend"
    )
    run.italic = True
    run.font.size = Pt(12)

    doc.add_paragraph()

    table(
        doc,
        ["Field", "Value"],
        [
            ["Repository", "ml-semester-project"],
            ["Backend", "Python 3.12 · FastAPI · psycopg 3 · scikit-learn"],
            ["Frontend", "Next.js 16 · TypeScript · Tailwind v4 · Recharts · next-intl"],
            ["Database", "PostgreSQL 18 · pgvector 0.8.7"],
            ["Audience", "Project author, supervisor, and anyone extending the platform"],
            ["Companion document", "Thesis-Final-Verified.docx (current v3 thesis)"],
        ],
        widths=[4.0, 11.5],
        font_size=10,
    )

    page_break(doc)


def build_how_to_use(doc) -> None:
    h1(doc, "0. How to Use This Guide")
    p(
        doc,
        "This document explains the platform as it is actually implemented. It "
        "is organised in the order in which the system executes, and within "
        "each layer it proceeds from the outermost structure inwards: the "
        "layer's responsibility, then its modules, then the individual "
        "functions with their signatures and the reasoning behind them.",
    )
    p(doc, "The guide has five parts.", bold_lead="")
    table(
        doc,
        ["Part", "Contents", "Read it when"],
        [
            [
                "1 — Orientation",
                "What the platform does, how the pieces fit, the repository map",
                "You are new to the project",
            ],
            [
                "2 — Environment",
                "Every setup step, in order, with the reasoning",
                "You are installing or rebuilding the environment",
            ],
            [
                "3 — Database",
                "pgvector installation, every table, every column, index lifecycle",
                "You are changing the schema or the retrieval strategy",
            ],
            [
                "4 — Backend",
                "Every module and every function of the FastAPI service",
                "You are changing the experimental logic",
            ],
            [
                "5 — Frontend",
                "Every file, component and helper of the Next.js application",
                "You are changing the user interface",
            ],
            [
                "6 — Appendix",
                "Data flow, extension recipes, troubleshooting",
                "You are integrating or debugging",
            ],
        ],
        widths=[2.8, 7.6, 5.1],
    )
    note(
        doc,
        "Every Python and TypeScript block in this guide is reproduced "
        "verbatim from the working implementation, and the set is checked "
        "automatically by docs/tools/verify_snippets.py, which compares every "
        "snippet against the source tree and reports any divergence. The "
        "shell commands, the SQL listings and the two diagrams are "
        "illustrative of what you type and of what PostgreSQL reports.",
    )


# ---------------------------------------------------------------------------
# Part 1
# ---------------------------------------------------------------------------
def build_orientation(doc) -> None:
    h1(doc, "1. Orientation")

    h2(doc, "1.1 What the Platform Does")
    p(
        doc,
        "The platform classifies tabular data with the K-Nearest Neighbors "
        "algorithm, but performs the neighbour retrieval inside PostgreSQL "
        "rather than in application memory. This is the central design choice, "
        "and it is what makes the project a study of vector similarity search "
        "rather than a study of KNN.",
    )
    p(
        doc,
        "Concretely, the platform can:",
    )
    bullet_list(
        doc,
        [
            "Ingest the UCI Forest CoverType dataset, or any labelled CSV "
            "uploaded by the user, and store the preprocessed feature vectors "
            "in pgvector tables.",
            "Run an exact retrieval baseline that is exhaustive by "
            "construction, and record its accuracy, macro F1 and per-query "
            "latency.",
            "Run an approximate retrieval using an HNSW index at any "
            "search-effort setting, and record the same metrics plus the "
            "retrieval recall against the exact baseline.",
            "Compare the two runs side by side, and plot the recall and "
            "accuracy curves across all runs of a dataset.",
        ],
    )

    h2(doc, "1.2 The One-Sentence Summary of the Architecture")
    p(
        doc,
        "The frontend configures experiments; the backend prepares the data "
        "and orchestrates the runs; PostgreSQL performs the retrieval. All "
        "vector mathematics happens in the database, so the comparison "
        "between exact and approximate search is a comparison between two "
        "access paths over identical data.",
    )

    h2(doc, "1.3 Repository Map")
    code(
        doc,
        "ml-semester-project/\n"
        "├── backend/                    Python service\n"
        "│   ├── app/\n"
        "│   │   ├── main.py            Application factory + lifespan\n"
        "│   │   ├── core/\n"
        "│   │   │   ├── config.py      Settings from the environment\n"
        "│   │   │   └── database.py   Connection pool + query helpers\n"
        "│   │   ├── db/\n"
        "│   │   │   ├── schema.py      The SQL schema, as a string\n"
        "│   │   │   └── init_db.py     Database creation + schema apply\n"
        "│   │   ├── api/\n"
        "│   │   │   ├── schemas.py     Pydantic request/response models\n"
        "│   │   │   ├── health.py      /api/health\n"
        "│   │   │   ├── datasets.py    Dataset ingestion endpoints\n"
        "│   │   │   └── experiments.py Experiment endpoints\n"
        "│   │   └── services/          The research logic (no HTTP knowledge)\n"
        "│   │       ├── preprocessing.py\n"
        "│   │       ├── datasets.py\n"
        "│   │       ├── metrics.py\n"
        "│   │       ├── vector_store.py    ← the core of the comparison\n"
        "│   │       ├── classification.py\n"
        "│   │       └── experiments.py\n"
        "│   ├── scripts/run_experiments.py\n"
        "│   ├── tests/                 47 pytest tests\n"
        "│   ├── requirements.txt\n"
        "│   └── .env.example\n"
        "│\n"
        "├── frontend/                   Next.js application\n"
        "│   ├── app/[locale]/          Route segments per language\n"
        "│   │   ├── layout.tsx         Root layout: fonts, providers, dir\n"
        "│   │   ├── page.tsx           Overview\n"
        "│   │   ├── datasets/          Dataset ingestion\n"
        "│   │   ├── experiments/       Experiment runner and results\n"
        "│   │   ├── compare/           Side-by-side comparison\n"
        "│   │   └── methodology/       Documentation of the method\n"
        "│   ├── components/            UI + feature components\n"
        "│   ├── i18n/                  routing.ts, request.ts, navigation.ts\n"
        "│   ├── lib/                   api.ts (client), utils.ts (formatting)\n"
        "│   ├── messages/              en.json, fa.json\n"
        "│   ├── providers/             Query + theme providers\n"
        "│   ├── types/api.ts           Shared API types\n"
        "│   └── proxy.ts               next-intl locale routing\n"
        "│\n"
        "├── docs/                       This guide, the thesis, the results\n"
        "│   └── tools/                 The generators for these documents\n"
        "├── pgvector/                   Locally built pgvector 0.8.7\n"
        "├── scripts/                    start-backend, start-frontend, verify\n"
        "└── README.md",
        caption="Listing 1 — The complete repository map.",
    )

    h2(doc, "1.4 Where the Important Logic Lives")
    table(
        doc,
        ["File", "Why it matters"],
        [
            [
                "backend/app/services/vector_store.py",
                "The exact / HNSW comparison. Everything the thesis measures "
                "originates here.",
            ],
            [
                "backend/app/services/experiments.py",
                "Composes one experiment: ground truth, measurement, "
                "classification, persistence.",
            ],
            [
                "backend/app/services/preprocessing.py",
                "The leakage contract. Fit and transform are separate functions "
                "for exactly this reason.",
            ],
            [
                "backend/app/db/schema.py",
                "The data model, including the per-dataset vector tables that "
                "make recall meaningful.",
            ],
            [
                "frontend/components/experiment-runner.tsx",
                "The control surface: every experiment parameter is set here.",
            ],
            [
                "frontend/lib/api.ts",
                "The single place that knows how to talk to the backend.",
            ],
        ],
        widths=[6.6, 8.9],
    )


# ---------------------------------------------------------------------------
# Part 2
# ---------------------------------------------------------------------------
def build_environment(doc) -> None:
    h1(doc, "2. Environment Setup")

    p(
        doc,
        "The following sequence produces a working environment. Each step "
        "records not only what to run but why it is needed, because three of "
        "the steps have non-obvious prerequisites.",
    )

    h2(doc, "2.1 PostgreSQL with pgvector")
    h3(doc, "2.1.1 Why the Extension Has to Be Built")
    p(
        doc,
        "The evaluation machine runs PostgreSQL 18 on Windows. pgvector does "
        "not ship a Windows binary, so it must be compiled from source. The "
        "source ships a Makefile.win target designed for exactly this case.",
    )
    p(
        doc,
        "Two obstacles follow. The first is the compiler: building a PostgreSQL "
        "extension requires the MSVC toolchain, which is present on this "
        "machine as part of the Visual Studio 2019 Build Tools. The second is "
        "installation: the PostgreSQL installation directory is not writable "
        "without administrator rights, and the PostgreSQL service runs as a "
        "restricted account.",
    )
    h3(doc, "2.1.2 Building")
    code(
        doc,
        "git clone --depth 1 --branch v0.8.7 https://github.com/pgvector/pgvector.git\n"
        "cd pgvector\n"
        "cmd /c \"call `\"C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\BuildTools\\\n"
        "         VC\\Auxiliary\\Build\\vcvars64.bat`\" &&\n"
        "         nmake /F Makefile.win PGROOT=`\"C:\\Program Files\\PostgreSQL\\18`\"\"",
        caption="Listing 2 — Building pgvector against the local PostgreSQL installation.",
    )
    p(
        doc,
        "The build produces vector.dll, vector.control and the SQL script. The "
        "PGROOT argument tells the makefile where the PostgreSQL headers and "
        "import library live.",
    )
    h3(doc, "2.1.3 Installing Without Administrator Rights")
    p(
        doc,
        "PostgreSQL 18 provides two GUCs that make a user-writable extension "
        "installation possible.",
    )
    table(
        doc,
        ["GUC", "Meaning", "Value used"],
        [
            [
                "extension_control_path",
                "Directories searched for .control and .sql extension files",
                "<project>\\pgvector",
            ],
            [
                "dynamic_library_path",
                "Directories searched for the extension shared library",
                "<project>\\pgvector/lib",
            ],
        ],
        widths=[4.6, 7.0, 3.9],
    )
    code(
        doc,
        "$PGVECTOR = \"<project>\\pgvector\"\n"
        "mkdir \"$PGVECTOR\\lib\", \"$PGVECTOR\\extension\"\n"
        "copy vector.dll             \"$PGVECTOR\\lib\\\"\n"
        "copy vector.control         \"$PGVECTOR\\extension\\\"\n"
        "copy sql\\vector--0.8.7.sql  \"$PGVECTOR\\extension\\\"",
        caption="Listing 3 — Placing the built artifacts.",
    )
    note(
        doc,
        "The extension sub-directory is not optional. When extension_control_path "
        "contains an entry other than $system, PostgreSQL appends /extension to "
        "it before searching. Placing the control file directly in the "
        "configured directory makes the extension invisible: pg_available_extensions "
        "returns nothing and CREATE EXTENSION reports that the extension is "
        "not available, with no further explanation.",
        label="Trap",
    )
    h3(doc, "2.1.4 Granting the Service Account Read Access")
    p(
        doc,
        "The PostgreSQL service runs as NT AUTHORITY\\NETWORK SERVICE. It "
        "cannot read files beneath a user profile by default, so the extension "
        "fails to load with “could not access file \\\"vector\\\"”. The fix is a "
        "read grant on the extension directory.",
    )
    code(
        doc,
        "icacls \"<project>\\pgvector\" /grant \"NT AUTHORITY\\NETWORK SERVICE:(OI)(CI)RX\" /T",
        caption="Listing 4 — Granting read access to the PostgreSQL service account.",
    )
    h3(doc, "2.1.5 Registering the Paths")
    code(
        doc,
        "psql -U postgres -h localhost -c \"ALTER SYSTEM SET extension_control_path = '<PGVECTOR>';\"\n"
        "psql -U postgres -h localhost -c \"ALTER SYSTEM SET dynamic_library_path = '<PGVECTOR>/lib';\"\n"
        "psql -U postgres -h localhost -c \"SELECT pg_reload_conf();\"\n"
        "\n"
        "# Verify\n"
        "psql -U postgres -h localhost -d postgres -c \"CREATE EXTENSION vector;\"\n"
        "psql -U postgres -h localhost -d postgres -c \"SELECT extversion FROM pg_extension WHERE extname='vector';\"",
        caption="Listing 5 — Registering the paths and verifying the installation.",
    )
    p(
        doc,
        "The verification query should print 0.8.7. The backend performs the "
        "same CREATE EXTENSION automatically when it starts, so the platform "
        "is ready as soon as the GUCs are in place.",
    )

    h2(doc, "2.2 Backend")
    code(
        doc,
        "cd backend\n"
        "python -m venv .venv\n"
        ".venv\\Scripts\\pip install -r requirements.txt\n"
        "Copy-Item .env.example .env\n"
        ".venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000",
        caption="Listing 6 — Installing and running the backend.",
    )
    table(
        doc,
        ["Requirement", "Purpose"],
        [
            ["fastapi", "HTTP framework, automatic OpenAPI documentation"],
            ["uvicorn[standard]", "ASGI server"],
            ["psycopg[binary,pool]", "PostgreSQL driver plus the connection pool"],
            ["pydantic / pydantic-settings", "Request validation and configuration"],
            ["numpy / pandas / scikit-learn", "Dataframe handling and standardisation"],
            ["httpx", "Dataset download over HTTP"],
            ["pytest / pytest-asyncio", "Test suite"],
        ],
        widths=[5.0, 10.5],
    )

    h2(doc, "2.3 Frontend")
    code(
        doc,
        "cd frontend\n"
        "bun install\n"
        "Copy-Item .env.example .env.local\n"
        "bun run dev",
        caption="Listing 7 — Installing and running the frontend.",
    )
    p(
        doc,
        "The .env.local file sets NEXT_PUBLIC_API_URL, the base address of the "
        "backend that the browser calls directly. The value is read at build "
        "time and inlined into the client bundle, which is why it must carry "
        "the NEXT_PUBLIC_ prefix.",
    )

    h2(doc, "2.4 Convenience Scripts")
    table(
        doc,
        ["Script", "Effect"],
        [
            [
                "scripts/start-backend.bat",
                "Creates the virtual environment if missing, copies .env if "
                "missing, then runs uvicorn with --reload",
            ],
            [
                "scripts/start-frontend.bat",
                "Installs dependencies if missing, copies .env.local if "
                "missing, then runs next dev",
            ],
            [
                "scripts/verify.bat",
                "Checks the pgvector extension, the API health endpoint, the "
                "frontend routes and the backend test suite in one pass",
            ],
        ],
        widths=[5.2, 10.3],
    )


# ---------------------------------------------------------------------------
# Part 3
# ---------------------------------------------------------------------------
def build_database(doc) -> None:
    h1(doc, "3. The Database Layer")

    h2(doc, "3.1 Connection Management")
    p(
        doc,
        "app/core/database.py owns every connection to PostgreSQL. The module "
        "exposes a single lazily-created pool and a small set of helpers.",
    )
    h3(doc, "3.1.1 get_pool()")
    code(
        doc,
        "def get_pool() -> ConnectionPool:\n"
        "    \"\"\"Return the lazily created global connection pool.\"\"\"\n"
        "    global _pool\n"
        "    if _pool is None:\n"
        "        _pool = ConnectionPool(\n"
        "            conninfo=settings.database_url,\n"
        "            min_size=settings.db_pool_min_size,\n"
        "            max_size=settings.db_pool_max_size,\n"
        "            kwargs={\"row_factory\": dict_row, \"autocommit\": True},\n"
        "            open=True,\n"
        "        )\n"
        "        _pool.wait(timeout=15)\n"
        "        logger.info(\"PostgreSQL connection pool ready at %s\", settings.postgres_db)\n"
        "    return _pool",
        caption="Listing 8 — The global connection pool.",
    )
    p(
        doc,
        "Two properties of this configuration matter downstream. The row "
        "factory is dict_row, so every query returns dictionaries keyed by "
        "column name; code that indexes a row positionally will fail, which is "
        "why the codebase uses r['id'] rather than r[0] throughout. And "
        "autocommit is enabled, which is fast for this workload but has the "
        "consequence documented in Section 4.5.2: SET LOCAL is silently "
        "discarded in autocommit mode.",
    )
    h3(doc, "3.1.2 The helper functions")
    table(
        doc,
        ["Function", "Signature", "Purpose"],
        [
            ["get_pool", "() -> ConnectionPool", "Lazily create and return the global pool"],
            [
                "connection",
                "() -> Iterator[Connection]",
                "Context manager yielding a pooled connection",
            ],
            ["fetch_all", "(sql, params) -> list[dict]", "Run a query, return every row"],
            ["fetch_one", "(sql, params) -> dict | None", "Run a query, return the first row"],
            ["execute", "(sql, params) -> None", "Run a statement, discard any result"],
            ["close_pool", "() -> None", "Shut the pool down at shutdown"],
        ],
        widths=[3.4, 5.6, 6.5],
    )
    note(
        doc,
        "The helpers always take a parameter tuple. Even though the project "
        "runs locally with a trusted user, parameterised queries are used "
        "throughout so that the code is correct against any deployment.",
    )

    h2(doc, "3.2 The Shared Tables")
    p(
        doc,
        "Three tables hold everything except the vectors themselves. Their "
        "definitions live in app/db/schema.py as a single SQL string, which "
        "migration_statements() splits into individual statements for "
        "idempotent application.",
    )

    h3(doc, "3.2.1 datasets")
    p(
        doc,
        "One row per ingested dataset. The row is the record of what the "
        "dataset is and, critically, of how it was transformed.",
    )
    table(
        doc,
        ["Column", "Type", "Purpose"],
        [
            ["id", "BIGSERIAL", "Primary key; also names the vector tables"],
            ["name", "TEXT", "Display name"],
            ["source", "TEXT", "'upload' or 'covertype'"],
            ["original_name", "TEXT", "The uploaded file name, if any"],
            ["n_rows", "INTEGER", "Rows in the source dataset"],
            ["n_features", "INTEGER", "Number of feature columns"],
            ["n_classes", "INTEGER", "Number of distinct classes"],
            ["dimension", "INTEGER", "Dimension of the stored vectors"],
            ["class_labels", "JSONB", "The class identifiers, in index order"],
            [
                "preprocessing",
                "JSONB",
                "The serialised fitted preprocessing state — the reproducibility record",
            ],
            ["is_seeded", "BOOLEAN", "True for the built-in Covertype dataset"],
            ["created_at / updated_at", "TIMESTAMPTZ", "Timestamps"],
        ],
        widths=[4.0, 2.6, 8.9],
    )
    note(
        doc,
        "The preprocessing column is what makes an experiment reproducible. It "
        "holds the fitted scaler means and scales, the categorical value "
        "mappings and the label mapping. A reader of the database can "
        "reconstruct exactly the transformation that produced any stored "
        "vector.",
    )

    h3(doc, "3.2.2 experiments")
    p(
        doc,
        "One row per experimental run. The table is deliberately wide: it "
        "stores the configuration, the metrics and the diagnostics together, "
        "so that a result can be interpreted without joining against anything.",
    )
    table(
        doc,
        ["Group", "Columns", "Purpose"],
        [
            [
                "Identity",
                "id, dataset_id, name, status, created_at, completed_at",
                "Identification and lifecycle",
            ],
            [
                "Configuration",
                "retrieval_method, k, distance_metric, hnsw_ef_search, "
                "hnsw_m, hnsw_ef_construction, repetitions, seed",
                "Everything needed to reproduce the run",
            ],
            [
                "Scale",
                "train_size, test_size, unique_queries, total_test_pool",
                "Pool sizes and the number of distinct queries evaluated",
            ],
            [
                "Provenance",
                "dataset_fingerprint, tie_tolerance_rel, tie_tolerance_abs",
                "Which exact pool composition was measured, and the tie tolerance",
            ],
            [
                "Retrieval quality",
                "recall_at_k, tie_aware_recall_at_k, tied_queries, "
                "max_tied_candidates, *_std",
                "Conventional and tie-aware overlap with the exact reference",
            ],
            [
                "Classification",
                "accuracy, macro_f1, precision_macro, recall_macro, "
                "accuracy_std, macro_f1_std",
                "The downstream effect of the retrieval",
            ],
            [
                "Performance",
                "latency_avg_ms, latency_p50_ms, latency_p95_ms, latency_p99_ms, "
                "latency_min_ms, latency_max_ms",
                "Per-query latency distribution over every stored observation",
            ],
            [
                "Index",
                "index_build_ms, index_created_ms, index_reused, "
                "index_reused_build_ms, index_size_bytes, used_index_scan",
                "Fresh build cost, reuse state and proof the index was used",
            ],
            [
                "Diagnostics",
                "confusion_matrix, per_class_metrics, config, environment",
                "JSONB detail for inspection and plotting",
            ],
            [
                "Sampling and order",
                "query_sampling, query_sample_seed, evaluated_query_ids, "
                "test_pool_class_distribution, evaluated_class_distribution, "
                "benchmark_round, benchmark_position, benchmark_seed, "
                "benchmark_order, warm_cache",
                "Which queries were evaluated and in what configuration order",
            ],
            [
                "Timing",
                "total_experiment_runtime_s, index_preparation_s, index_build_s, "
                "ground_truth_computation_s, retrieval_total_s, "
                "retrieval_mean_per_query_ms, metrics_computation_s, "
                "persistence_s, orchestration_overhead_s",
                "Non-overlapping perf_counter intervals for the run",
            ],
        ],
        widths=[2.8, 7.2, 5.5],
    )
    p(
        doc,
        "Two column groups deserve emphasis. used_index_scan records whether the "
        "planner actually chose an HNSW index scan for the measured query; "
        "without it, a silent fallback to a sequential scan would be "
        "indistinguishable from a legitimate approximate result. The "
        "index_build_ms / index_reused / index_reused_build_ms trio separates a "
        "build that happened during this run from a historical build that is "
        "being reused: on a reused index the current build time is NULL and the "
        "historical duration lives in index_reused_build_ms. The timing columns "
        "replace the retired api_overhead_s, which was total runtime minus "
        "query time and therefore charged ground-truth computation and "
        "orchestration to 'API overhead'. The legacy total_runtime_s is kept as "
        "an alias for older clients.",
    )

    h3(doc, "3.2.3 query_latencies")
    p(
        doc,
        "One row per measured query, per repetition: the experiment id, the "
        "repetition number, the query index, the latency in milliseconds and "
        "the identifiers that were retrieved. Storing the individual samples "
        "rather than only the summary percentiles is what allows the interface "
        "to draw a latency histogram, to report the p90, and to compute a "
        "between-repetition spread. The (experiment_id, repetition, query_index) "
        "triple identifies a sample uniquely.",
    )
    code(
        doc,
        "CREATE TABLE IF NOT EXISTS query_latencies (\n"
        "    id            BIGSERIAL PRIMARY KEY,\n"
        "    experiment_id BIGINT      NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,\n"
        "    repetition    INTEGER     NOT NULL DEFAULT 1,\n"
        "    query_index   INTEGER     NOT NULL,\n"
        "    latency_ms    DOUBLE PRECISION NOT NULL,\n"
        "    retrieved_ids JSONB       NOT NULL DEFAULT '[]'::jsonb\n"
        ");",
        caption="Listing 9 — The per-query latency table.",
    )

    h3(doc, "3.2.4 experiment_repetitions")
    p(
        doc,
        "One row per experiment and repetition, holding that pass's latency "
        "distribution and its quality metrics. A repetition is a complete timed "
        "pass over the whole query set, so its recall and accuracy describe that "
        "pass rather than an average over all latency samples. These rows are "
        "what make the between-repetition spread measurable, and they are "
        "written from the measured neighbour sets rather than recomputed from "
        "the aggregate.",
    )
    code(
        doc,
        "CREATE TABLE IF NOT EXISTS experiment_repetitions (\n"
        "    id                   BIGSERIAL PRIMARY KEY,\n"
        "    experiment_id        BIGINT      NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,\n"
        "    repetition           INTEGER     NOT NULL,\n"
        "    unique_queries       INTEGER     NOT NULL,\n"
        "    samples              INTEGER     NOT NULL,\n"
        "    latency_avg_ms       DOUBLE PRECISION,\n"
        "    latency_p50_ms       DOUBLE PRECISION,\n"
        "    latency_p95_ms       DOUBLE PRECISION,\n"
        "    latency_p99_ms       DOUBLE PRECISION,\n"
        "    recall_at_k          DOUBLE PRECISION,\n"
        "    tie_aware_recall_at_k DOUBLE PRECISION,\n"
        "    accuracy             DOUBLE PRECISION,\n"
        "    precision_macro      DOUBLE PRECISION,\n"
        "    recall_macro         DOUBLE PRECISION,\n"
        "    macro_f1             DOUBLE PRECISION,\n"
        "    UNIQUE (experiment_id, repetition)\n"
        ");",
        caption="Listing 10 — The per-repetition result table.",
    )

    h3(doc, "3.2.5 hnsw_indexes")
    p(
        doc,
        "A registry that records the metric and build parameters of the index "
        "that currently exists for each dataset. It exists for one reason: "
        "building an HNSW index costs seconds, and ef_search is a query-time "
        "parameter, so a sweep over many ef_search values must not rebuild the "
        "index each time.",
    )
    code(
        doc,
        "CREATE TABLE IF NOT EXISTS hnsw_indexes (\n"
        "    dataset_id      BIGINT      PRIMARY KEY REFERENCES datasets(id) ON DELETE CASCADE,\n"
        "    distance_metric TEXT        NOT NULL,\n"
        "    m               INTEGER     NOT NULL,\n"
        "    ef_construction INTEGER     NOT NULL,\n"
        "    build_ms        DOUBLE PRECISION,\n"
        "    size_bytes      BIGINT,\n"
        "    built_at        TIMESTAMPTZ NOT NULL DEFAULT now()\n"
        ");",
        caption="Listing 11 — The HNSW index registry.",
    )

    h2(doc, "3.3 The Per-Dataset Vector Tables")
    p(
        doc,
        "Each dataset owns two tables, named fv_<dataset_id> (the training "
        "pool) and tv_<dataset_id> (the held-out query pool). They are created "
        "when the dataset is ingested and dropped when it is deleted.",
    )
    code(
        doc,
        "CREATE TABLE fv_1 (\n"
        "    id        BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,\n"
        "    row_index INTEGER NOT NULL,\n"
        "    label     SMALLINT NOT NULL,\n"
        "    embedding vector(54) NOT NULL\n"
        ");\n"
        "CREATE INDEX fv_1_row_idx ON fv_1 (row_index);",
        caption="Listing 12 — The training-pool table for a 54-dimensional dataset, as created in the database.",
    )
    table(
        doc,
        ["Column", "Type", "Purpose"],
        [
            [
                "id",
                "BIGINT identity",
                "Primary key. This is the identifier that Recall@K compares.",
            ],
            [
                "row_index",
                "INTEGER",
                "The row number in the source frame, kept for traceability",
            ],
            ["label", "SMALLINT", "The class index of the observation"],
            [
                "embedding",
                "vector(54)",
                "The preprocessed feature vector, with an explicit dimension",
            ],
        ],
        widths=[3.0, 3.4, 9.1],
    )

    h3(doc, "3.3.1 Why Not One Shared Table?")
    p(
        doc,
        "A single table with a dataset_id column is the obvious design, and it "
        "is wrong in two independent ways. Both were discovered empirically "
        "during implementation.",
    )
    h4(doc, "Reason one — the index needs a fixed dimension")
    p(
        doc,
        "Creating an HNSW index over an unconstrained vector column fails:",
    )
    code(
        doc,
        "CREATE INDEX fv_embedding_hnsw_idx ON fv USING hnsw (embedding vector_l2_ops);\n"
        "ERROR:  column does not have dimensions",
        caption="Listing 13 — The error that forces a dimensionally-typed column.",
    )
    p(
        doc,
        "A shared table would need one column per possible dimensionality. "
        "Separate tables give each dataset exactly the column type it needs.",
    )
    h4(doc, "Reason two — residual filters are applied after the index scan")
    p(
        doc,
        "pgvector's HNSW scan gathers candidate neighbours from the whole "
        "table and only then applies the WHERE clause. On a shared table a "
        "filtered query therefore returns neighbours from the wrong dataset. "
        "This is silent: the query succeeds, returns plausible rows, and "
        "corrupts Recall@K without any error.",
    )
    note(
        doc,
        "This is the failure mode that would have quietly invalidated the "
        "entire experimental design had it not been caught. It is worth "
        "stating plainly: the per-dataset table is not a stylistic "
        "preference, it is a correctness requirement.",
        label="Important",
    )

    h2(doc, "3.4 The HNSW Index")
    code(
        doc,
        "CREATE INDEX fv_1_embedding_hnsw_idx ON fv_1\n"
        "    USING hnsw (embedding vector_l2_ops)\n"
        "    WITH (m = 16, ef_construction = 64);",
        caption="Listing 14 — The statement the platform issues to build the index, as recorded by PostgreSQL.",
    )
    table(
        doc,
        ["Parameter", "Scope", "Effect"],
        [
            [
                "m",
                "Build",
                "Graph degree. Higher values improve recall and increase "
                "construction and search cost.",
            ],
            [
                "ef_construction",
                "Build",
                "Size of the candidate list used while building. Higher values "
                "improve index quality at the cost of build time.",
            ],
            [
                "ef_search",
                "Query",
                "Size of the candidate list used while searching. The knob "
                "this project studies.",
            ],
            [
                "vector_l2_ops / vector_cosine_ops / vector_l1_ops",
                "Build",
                "The operator class. It must match the distance operator used "
                "in the query, or PostgreSQL rejects the query.",
            ],
        ],
        widths=[4.6, 2.0, 8.9],
    )
    p(
        doc,
        "Changing m or ef_construction requires a rebuild; changing ef_search "
        "does not. The registry table in Section 3.2.5 exists so that the "
        "platform can tell the difference and reuse the index whenever the "
        "build parameters are unchanged.",
    )

    h2(doc, "3.5 Schema Bootstrap")
    p(
        doc,
        "app/db/init_db.py runs at application startup, inside the lifespan "
        "handler.",
    )
    table(
        doc,
        ["Function", "What it does"],
        [
            [
                "ensure_database_exists()",
                "Connects to the maintenance database and issues CREATE "
                "DATABASE if the application database is missing. Uses the "
                "admin_database_url property, which points at the postgres "
                "database rather than the application one.",
            ],
            [
                "init_schema()",
                "Splits SCHEMA_SQL into statements with migration_statements() "
                "and executes each one. Every statement uses IF NOT EXISTS, so "
                "the function is idempotent and safe to call on every start.",
            ],
        ],
        widths=[5.4, 10.1],
    )
    p(
        doc,
        "The vector tables are not part of SCHEMA_SQL, because their "
        "dimensionality depends on the dataset. They are created "
        "individually by create_dataset_tables() during ingestion.",
    )


# ---------------------------------------------------------------------------
# Part 4
# ---------------------------------------------------------------------------
def build_backend(doc) -> None:
    h1(doc, "4. The Backend")

    p(
        doc,
        "The backend follows a strict two-layer split. The api package "
        "translates between HTTP and Python; the services package contains "
        "the experimental logic and knows nothing about HTTP. A service "
        "function can therefore be called from a script, a test or a notebook "
        "without any HTTP scaffolding.",
    )

    h2(doc, "4.1 Application Entry Point")
    p(doc, "app/main.py", bold_lead="")
    code(
        doc,
        "@asynccontextmanager\n"
        "async def lifespan(app: FastAPI):\n"
        "    logger.info(\"Starting KNN classification platform backend\")\n"
        "    ensure_database_exists()\n"
        "    init_schema()\n"
        "    yield\n"
        "    close_pool()\n"
        "    logger.info(\"Backend shut down\")",
        caption="Listing 15 — The lifespan handler.",
    )
    p(
        doc,
        "The database and schema are therefore guaranteed to exist before the "
        "first request is served, and the pool is closed cleanly on shutdown. "
        "CORS is configured from settings.cors_origins, which by default "
        "permits localhost:3000 and 127.0.0.1:3000 — the origins the "
        "frontend dev server uses.",
    )

    h2(doc, "4.2 Configuration")
    p(doc, "app/core/config.py", bold_lead="")
    table(
        doc,
        ["Setting", "Default", "Purpose"],
        [
            ["postgres_host / _port / _user / _password", "localhost / 5432 / postgres / …", "Connection parameters"],
            ["postgres_db", "knn_knn_platform", "The application database"],
            ["db_pool_min_size / _max_size", "2 / 10", "Pool bounds"],
            ["db_command_timeout_s", "120", "Command timeout"],
            ["api_host / _port / _prefix", "127.0.0.1 / 8000 / /api", "API binding and path prefix"],
            ["cors_origins", "localhost:3000, 127.0.0.1:3000", "Permitted browser origins"],
            ["data_dir", "backend/data", "Where the dataset cache lives"],
            ["covertype_url", "UCI archive URL", "Where the seeded dataset is fetched from"],
            ["default_k", "5", "Default neighbourhood size"],
            ["default_test_size", "5,000", "Default held-out size"],
            ["max_test_size", "50,000", "Upper bound on the held-out size"],
        ],
        widths=[6.2, 4.6, 4.7],
    )
    p(
        doc,
        "Two derived properties build the connection strings. database_url "
        "points at the application database and is what the pool uses; "
        "admin_database_url points at the postgres maintenance database and is "
        "what ensure_database_exists() uses, because the application database "
        "does not exist yet at that moment.",
    )

    h2(doc, "4.3 Preprocessing")
    p(doc, "app/services/preprocessing.py", bold_lead="")
    p(
        doc,
        "This module contains the leakage contract. The rule is simple and "
        "structural: fitting and applying are separate operations, and the "
        "fit operation only ever sees training data.",
    )
    table(
        doc,
        ["Function", "Purpose"],
        [
            [
                "infer_column_types(df, target_column)",
                "Splits the feature columns into numeric and categorical groups.",
            ],
            [
                "_label_key(value)",
                "Normalises a raw class label into a stable integer key, "
                "handling numeric, integral-float and string labels.",
            ],
            [
                "fit_preprocessor(df, target_column)",
                "Learns the scaler parameters and the categorical mappings from "
                "the given frame. Returns the state and the fitted scaler.",
            ],
            [
                "transform(df, state, scaler)",
                "Applies an already-fitted state to any frame and returns the "
                "scaled matrix and the positionally encoded labels.",
            ],
            [
                "vector_literal(matrix)",
                "Converts a float matrix into pgvector text literals such as "
                "[0.12,-0.34,...] for bulk loading.",
            ],
        ],
        widths=[5.8, 9.7],
    )
    h3(doc, "4.3.1 PreprocessState")
    p(
        doc,
        "The dataclass holds everything that was learned: the feature column "
        "list, the numeric and categorical groupings, the category-to-index "
        "mappings, the scaler means and scales, the class labels and the "
        "label-to-index mapping. It serialises to JSON, and from_json() "
        "coerces the label mapping keys back to integers, because JSON object "
        "keys are always strings.",
    )
    code(
        doc,
        "@classmethod\n"
        "def from_dict(cls, data: dict[str, Any]) -> PreprocessState:\n"
        "    state = cls(**data)\n"
        "    # JSON object keys are always strings, so the label mapping has to be\n"
        "    # coerced back to integers.\n"
        "    state.label_to_index = {int(k): int(v) for k, v in state.label_to_index.items()}\n"
        "    state.class_labels = [int(v) for v in state.class_labels]\n"
        "    return state",
        caption="Listing 16 — Deserialising the preprocessing state.",
    )
    h3(doc, "4.3.2 A Note on Label Encoding")
    p(
        doc,
        "The labels stored in the database are positional indices 0..n-1, while "
        "the class_labels list holds the original label values for display. "
        "evaluate_predictions therefore treats its inputs positionally and "
        "uses class_labels only for presentation. Mixing the two conventions "
        "would produce a confusion matrix whose axes were silently mislabelled.",
    )

    h2(doc, "4.4 Distance Metrics")
    p(doc, "app/services/metrics.py", bold_lead="")
    p(
        doc,
        "DistanceMetric is a StrEnum with three members. Each exposes the "
        "pgvector operator, the pgvector function and a human-readable label.",
    )
    table(
        doc,
        ["Member", "Operator", "Function", "Operator class"],
        [
            ["L2", "<->", "l2_distance", "vector_l2_ops"],
            ["COSINE", "<=>", "cosine_distance", "vector_cosine_ops"],
            ["L1", "<+>", "l1_distance", "vector_l1_ops"],
        ],
        widths=[3.0, 3.4, 4.6, 4.5],
    )
    p(
        doc,
        "The operator and the function are mathematically equivalent but not "
        "planner-equivalent, and that difference is the basis of the whole "
        "experimental design. See Section 4.5.1.",
    )

    h2(doc, "4.5 The Vector Store — the Core of the Project")
    p(doc, "app/services/vector_store.py", bold_lead="")
    p(
        doc,
        "This module owns the per-dataset tables, the HNSW index, and the two "
        "retrieval strategies. It is the file to read first.",
    )

    h3(doc, "4.5.1 Table and Index Naming")
    table(
        doc,
        ["Function", "Returns", "Purpose"],
        [
            ["train_table(dataset_id)", "'fv_<id>'", "The training-pool table name"],
            ["test_table(dataset_id)", "'tv_<id>'", "The query-pool table name"],
            ["index_name(dataset_id)", "'fv_<id>_embedding_hnsw_idx'", "The HNSW index name"],
        ],
        widths=[5.6, 4.8, 5.1],
    )
    p(
        doc,
        "These names are computed, never stored, so there is exactly one "
        "place in the codebase that decides how a dataset maps onto physical "
        "objects.",
    )

    h3(doc, "4.5.2 Table Lifecycle")
    code(
        doc,
        "def create_dataset_tables(dataset_id: int, dimension: int) -> None:\n"
        "    \"\"\"Create the train/test tables for a dataset with a typed vector column.\"\"\"\n"
        "    dim = int(dimension)\n"
        "    with connection() as conn:\n"
        "        with conn.cursor() as cur:\n"
        "            for table in (train_table(dataset_id), test_table(dataset_id)):\n"
        "                cur.execute(sql.SQL(\"DROP TABLE IF EXISTS {t} CASCADE\").format(\n"
        "                    t=sql.Identifier(table)\n"
        "                ))\n"
        "                cur.execute(\n"
        "                    sql.SQL(\n"
        "                        \"CREATE TABLE {t} (\"\n"
        "                        \"  id          BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,\"\n"
        "                        \"  row_index   INTEGER NOT NULL,\"\n"
        "                        \"  label       SMALLINT NOT NULL,\"\n"
        "                        \"  embedding   vector({dim}) NOT NULL\"\n"
        "                        \")\"\n"
        "                    ).format(t=sql.Identifier(table), dim=sql.Literal(dim))\n"
        "                )\n"
        "                cur.execute(\n"
        "                    sql.SQL(\"CREATE INDEX {i} ON {t} (row_index)\").format(\n"
        "                        i=sql.Identifier(f\"{table}_row_idx\"), t=sql.Identifier(table)\n"
        "                    )\n"
        "                )\n"
        "    logger.info(\"Created vector tables for dataset %s (dim=%s)\", dataset_id, dim)",
        caption="Listing 17 — Creating the vector tables with a typed dimension.",
    )
    p(
        doc,
        "Every identifier is interpolated with psycopg's sql.Identifier and "
        "sql.Literal rather than by string formatting. This is not only about "
        "safety: it is about correctness. A plain Python .format() with a "
        "sql.Identifier renders the literal text Identifier('fv_1'), which "
        "PostgreSQL rejects with “function identifier(unknown) does not exist”. "
        "Only the sql.SQL(...).format(...) composition produces a properly "
        "quoted identifier.",
    )
    note(
        doc,
        "Note that sql.Literal(dim) is used for the dimension. The dimension "
        "is an integer that reaches the SQL as a literal rather than as a "
        "parameter, because a vector(54) type modifier cannot be expressed as "
        "a bind parameter.",
        label="Note",
    )

    h3(doc, "4.5.3 Bulk Loading")
    code(
        doc,
        "def insert_vectors(\n"
        "    dataset_id: int,\n"
        "    which: str,\n"
        "    embeddings: Sequence[str],\n"
        "    labels: Sequence[int],\n"
        "    row_indices: Sequence[int],\n"
        ") -> int:\n"
        "    \"\"\"Bulk insert feature vectors using COPY for throughput.\"\"\"\n"
        "    table = train_table(dataset_id) if which == \"train\" else test_table(dataset_id)\n"
        "\n"
        "    with connection() as conn:\n"
        "        with conn.cursor() as cur:\n"
        "            with cur.copy(\n"
        "                sql.SQL(\n"
        "                    \"COPY {table} (row_index, label, embedding) FROM STDIN\"\n"
        "                ).format(table=sql.Identifier(table))\n"
        "            ) as copy:\n"
        "                for emb, lab, idx in zip(embeddings, labels, row_indices):\n"
        "                    copy.write_row((int(idx), int(lab), emb))\n"
        "    logger.info(\"Inserted %s rows into %s\", len(labels), table)\n"
        "    return len(labels)",
        caption="Listing 18 — Bulk-loading vectors with COPY.",
    )
    p(
        doc,
        "COPY is used rather than INSERT because the load is a bulk operation: "
        "for the 50,000-row training pool this is the difference between a "
        "few seconds and a few minutes. The COPY statement uses the default "
        "text format with tab delimiters. An earlier attempt used "
        "FORMAT csv, which fails with “extra data after last expected column” "
        "because the vector literal contains commas that the CSV parser then "
        "treats as column separators.",
    )

    h3(doc, "4.5.4 Index Management")
    table(
        doc,
        ["Function", "Signature", "Purpose"],
        [
            [
                "drop_hnsw_index",
                "(dataset_id)",
                "Drops the index for a dataset",
            ],
            [
                "ensure_hnsw_index",
                "(dataset_id, metric, m, ef_construction, force) -> IndexInfo",
                "Builds the index or reuses a compatible one, and reports "
                "which of the two happened",
            ],
            [
                "create_hnsw_index",
                "(dataset_id, metric, m, ef_construction, force) -> float",
                "Backward-compatible wrapper returning the build time, or 0.0 "
                "when the index was reused",
            ],
            [
                "get_hnsw_index",
                "(dataset_id) -> dict | None",
                "Returns the registered build parameters, if any",
            ],
            [
                "hnsw_index_exists",
                "(dataset_id) -> bool",
                "Checks pg_indexes for the physical index",
            ],
            [
                "index_size_bytes",
                "(dataset_id) -> int | None",
                "Reads pg_relation_size for the live index",
            ],
            [
                "analyze",
                "(dataset_id)",
                "Refreshes planner statistics on both tables",
            ],
        ],
        widths=[4.4, 6.2, 4.9],
    )
    p(
        doc,
        "ensure_hnsw_index is the function that contains the reuse logic. It "
        "consults the registry table and, if the metric and build parameters "
        "match and the physical index is still present, returns an IndexInfo "
        "with created = False. Otherwise it builds the index and returns "
        "created = True with the measured duration. The returned parameters are "
        "read back from the live index rather than copied from the Python "
        "defaults, so a persisted row describes what PostgreSQL actually built.",
    )
    code(
        doc,
        "existing = get_hnsw_index(dataset_id)\n"
        "if (\n"
        "    not force\n"
        "    and existing is not None\n"
        "    and existing[\"distance_metric\"] == metric.value\n"
        "    and int(existing[\"m\"]) == int(m)\n"
        "    and int(existing[\"ef_construction\"]) == int(ef_construction)\n"
        "    and hnsw_index_exists(dataset_id)\n"
        "):\n"
        "    logger.info(\"Reusing existing HNSW index for dataset %s\", dataset_id)\n"
        "    return IndexInfo(\n"
        "        created=False,\n"
        "        build_ms=float(existing[\"build_ms\"]) if existing[\"build_ms\"] is not None else None,\n"
        "        m=int(existing[\"m\"]),\n"
        "        ef_construction=int(existing[\"ef_construction\"]),\n"
        "        size_bytes=index_size_bytes(dataset_id),\n"
        "        distance_metric=str(existing[\"distance_metric\"]),\n"
        "    )",
        caption="Listing 19 — The index reuse check.",
    )
    note(
        doc,
        "The exact baseline deliberately does not drop the index. Because the "
        "exact query uses the distance function, the index cannot be used by "
        "it anyway, so leaving the index in place means the next HNSW run "
        "reuses it instead of paying the build cost again.",
    )

    h3(doc, "4.5.5 The Two Query Shapes")
    code(
        doc,
        "def _distance_expr(metric: DistanceMetric, use_operator: bool) -> sql.Composable:\n"
        "    \"\"\"Distance expression for the metric.\n"
        "\n"
        "    The *operator* form (``embedding <-> %s``) is what pgvector's HNSW index\n"
        "    can satisfy; the *function* form (``l2_distance(...)``) always forces a\n"
        "    full sequential scan and therefore serves as the exact baseline.\n"
        "    \"\"\"\n"
        "    if use_operator:\n"
        "        return sql.SQL(\"embedding {} %s\").format(sql.SQL(metric.operator))\n"
        "    return sql.SQL(\"{} (embedding, %s::vector)\").format(sql.SQL(metric.function))",
        caption="Listing 20 — Choosing between the operator form and the function form.",
    )
    p(
        doc,
        "The distance expression appears twice in the query, once in the "
        "projection and once in the ordering, so the query vector is supplied "
        "twice in the parameter tuple. The alternative of computing the "
        "distance once in a CTE changes the plan shape and was rejected "
        "because it risks the planner losing the index ordering.",
    )
    code(
        doc,
        "def _order_expr(dist: sql.Composable, use_operator: bool) -> sql.Composable:\n"
        "    \"\"\"Ordering for a nearest-neighbour query.\n"
        "\n"
        "    The exhaustive path gets an ``id`` tiebreak so that a repeated exact query\n"
        "    returns the same identifiers even when several rows share a distance.  The\n"
        "    HNSW path keeps only the index-compatible distance ordering: adding a\n"
        "    secondary sort key makes the planner abandon the index scan, which would\n"
        "    turn an approximate measurement into a sequential scan.\n"
        "    \"\"\"\n"
        "    if use_operator:\n"
        "        return dist\n"
        "    return sql.SQL(\"{dist}, id\").format(dist=dist)\n"
        "\n"
        "\n"
        "def _query_sql(\n"
        "    dataset_id: int, metric: DistanceMetric, k: int, use_operator: bool\n"
        ") -> sql.Composed:\n"
        "    dist = _distance_expr(metric, use_operator)\n"
        "    order = _order_expr(dist, use_operator)\n"
        "    return sql.SQL(\n"
        "        \"SELECT id, label, {dist} AS distance \"\n"
        "        \"FROM {t} \"\n"
        "        \"ORDER BY {order} \"\n"
        "        \"LIMIT {k}\"\n"
        "    ).format(\n"
        "        dist=dist,\n"
        "        order=order,\n"
        "        t=sql.Identifier(train_table(dataset_id)),\n"
        "        k=sql.Literal(k),\n"
        "    )\n"
        "\n"
        "\n"
        "def _explain_sql(\n"
        "    dataset_id: int, metric: DistanceMetric, k: int, use_operator: bool\n"
        ") -> sql.Composed:\n"
        "    dist = _distance_expr(metric, use_operator)\n"
        "    order = _order_expr(dist, use_operator)\n"
        "    return sql.SQL(\n"
        "        \"EXPLAIN (ANALYZE, BUFFERS, FORMAT TEXT) \"\n"
        "        \"SELECT id, label, {dist} AS distance \"\n"
        "        \"FROM {t} \"\n"
        "        \"ORDER BY {order} \"\n"
        "        \"LIMIT {k}\"\n"
        "    ).format(\n"
        "        dist=dist,\n"
        "        order=order,\n"
        "        t=sql.Identifier(train_table(dataset_id)),\n"
        "        k=sql.Literal(k),\n"
        "    )\n"
        "\n"
        "",
        caption="Listing 21 — The retrieval query builder.",
    )

    h3(doc, "4.5.6 retrieve() — The Measured Function")
    code(
        doc,
        "def retrieve(\n"
        "    query_vector: str,\n"
        "    k: int,\n"
        "    metric: DistanceMetric,\n"
        "    dataset_id: int,\n"
        "    method: str = \"exact\",\n"
        "    ef_search: int | None = None,\n"
        "    use_operator: bool = True,\n"
        "    warmup: bool = True,\n"
        "    verify_plan: bool = True,\n"
        ") -> RetrievalResult:\n"
        "    \"\"\"Run a single nearest-neighbour query and time it.\n"
        "\n"
        "    ``method='exact'`` uses the distance *function* form, which can never be\n"
        "    satisfied by an index and therefore guarantees an exhaustive scan.\n"
        "    ``method='hnsw'`` uses the *operator* form so the planner can choose an\n"
        "    HNSW index scan.\n"
        "\n"
        "    The search-effort parameter is set at *session* level rather than with\n"
        "    ``SET LOCAL``, because the connection pool runs in autocommit mode where\n"
        "    ``SET LOCAL`` is silently discarded. The value is read back from the\n"
        "    server to confirm that it was actually applied, and it is restored in a\n"
        "    ``finally`` block so that a pooled connection never leaks the setting\n"
        "    into a later request.\n"
        "    \"\"\"\n"
        "    use_op = method == \"hnsw\" and use_operator\n"
        "    applied_ef: int | None = None\n"
        "\n"
        "    with connection() as conn:\n"
        "        with conn.cursor() as cur:\n"
        "            if ef_search is not None:\n"
        "                cur.execute(f\"SET hnsw.ef_search = {int(ef_search)}\")\n"
        "                # Confirm the server actually took the value.\n"
        "                cur.execute(\"SHOW hnsw.ef_search\")\n"
        "                row = cur.fetchone()\n"
        "                applied_ef = int(row[\"hnsw.ef_search\"]) if row else None\n"
        "            try:\n"
        "                stmt = _query_sql(dataset_id, metric, k, use_op)\n"
        "                params: tuple = (query_vector, query_vector)\n"
        "\n"
        "                # Warm-up run so that the measured latency is not dominated by\n"
        "                # first-call planning and buffer-cache population.\n"
        "                if warmup:\n"
        "                    cur.execute(stmt, params)\n"
        "                    cur.fetchall()\n"
        "\n"
        "                start = time.perf_counter()\n"
        "                cur.execute(stmt, params)\n"
        "                rows = cur.fetchall()\n"
        "                elapsed_ms = (time.perf_counter() - start) * 1000\n"
        "\n"
        "                plan = \"\"\n"
        "                used_index = False\n"
        "                used_seq_scan = False\n"
        "                if verify_plan:\n"
        "                    cur.execute(_explain_sql(dataset_id, metric, k, use_op), params)\n"
        "                    plan = \"\\n\".join(r[\"QUERY PLAN\"] for r in cur.fetchall())\n"
        "                    used_index = plan_uses_index_scan(plan, index_name(dataset_id))\n"
        "                    used_seq_scan = plan_uses_sequential_scan(plan)\n"
        "            finally:\n"
        "                if ef_search is not None:\n"
        "                    cur.execute(\"RESET hnsw.ef_search\")\n"
        "\n"
        "    neighbours = [\n"
        "        Neighbour(\n"
        "            row_id=int(r[\"id\"]),\n"
        "            label=int(r[\"label\"]),\n"
        "            distance=float(r[\"distance\"]),\n"
        "        )\n"
        "        for r in rows\n"
        "    ]\n"
        "    return RetrievalResult(\n"
        "        neighbours=neighbours,\n"
        "        latency_ms=elapsed_ms,\n"
        "        used_index_scan=used_index,\n"
        "        plan=plan,\n"
        "        used_sequential_scan=used_seq_scan,\n"
        "        applied_ef_search=applied_ef,\n"
        "    )",
        caption="Listing 22 — The measured retrieval function, in full.",
    )
    p(
        doc,
        "Seven decisions in this function carry the experimental validity:",
    )
    numbered(
        doc,
        "use_op is false for the exact method, which selects the distance "
        "function and therefore guarantees an exhaustive scan.",
    )
    numbered(
        doc,
        "The query vector appears twice in the parameter tuple because the "
        "distance expression appears twice in the SQL.",
    )
    numbered(
        doc,
        "The warm-up query is executed and discarded, so the measured "
        "latency excludes first-call planning and cold-buffer effects.",
    )
    numbered(
        doc,
        "The timer brackets only the measured execution, not the "
        "parameterisation, the cursor handling or the result transfer.",
    )
    numbered(
        doc,
        "The ef_search setting is read back with SHOW and returned as "
        "applied_ef_search, so a silently ignored SET cannot be mistaken for a "
        "real sweep.",
    )
    numbered(
        doc,
        "Both access-path flags are recorded: used_index_scan for HNSW and "
        "used_sequential_scan for the exact baseline, so each method's plan is "
        "proven rather than assumed.",
    )
    numbered(
        doc,
        "The ef_search setting is restored in a finally block, so a pooled "
        "connection never carries the setting into another request.",
    )
    note(
        doc,
        "The SET LOCAL trap deserves repeating because it silently invalidates "
        "the experiment. In autocommit mode PostgreSQL warns “SET LOCAL can "
        "only be used in transaction blocks” and discards the setting. An "
        "early sweep used SET LOCAL, so every run silently used the default "
        "ef_search of 40 and the recall curve was flat and noisy. The "
        "symptom was not an error but an implausible result, which is why "
        "the parameter is now set at session level and reset explicitly.",
        label="Trap",
    )

    h3(doc, "4.5.7 The Ground Truth")
    p(
        doc,
        "ground_truth_candidates() issues the exact query for every query "
        "vector and returns a GroundTruth object per query. It always uses the "
        "function form, regardless of the method under test, so the reference "
        "is exact by construction. Two properties make it reproducible:",
    )
    bullet_list(
        doc,
        [
            "The exact top-k is ordered by (distance, id). Without the id "
            "tiebreak the K-th row would be whatever the executor happened to "
            "return, and conventional Recall@K would not be reproducible.",
            "A second query returns every id whose distance is within the "
            "documented tolerance of the K-th boundary, "
            "boundary × (1 + rel) + abs, so a returned neighbour that is "
            "equally distant but differently identified can be recognised as "
            "valid rather than counted as a miss.",
        ],
    )
    p(
        doc,
        "The object exposes is_tied and candidate_ids, so both recall "
        "definitions are computed from the same measurement rather than from "
        "two different queries. The tolerance is a TieTolerance value stored on "
        "the experiment row.",
    )

    h3(doc, "4.5.8 Plan Verification")
    p(
        doc,
        "Once per batch the retrieval query is re-executed under EXPLAIN "
        "(ANALYZE, FORMAT TEXT), and the plan text is tested by "
        "plan_uses_index_scan(). When the index name is supplied, the plan must "
        "reference that specific index, because a btree index scan would "
        "otherwise satisfy the generic check and produce a false positive.",
    )
    code(
        doc,
        "def plan_uses_index_scan(plan_text: str, index_name_: str | None = None) -> bool:\n"
        "    \"\"\"Detect whether the planner chose an HNSW index scan.\n"
        "\n"
        "    When ``index_name_`` is given, the plan must reference that specific\n"
        "    index. This matters because a btree index scan would otherwise satisfy\n"
        "    the generic check and produce a false positive.\n"
        "    \"\"\"\n"
        "    lowered = plan_text.lower()\n"
        "    if \"index scan\" not in lowered:\n"
        "        return False\n"
        "    if index_name_ is not None:\n"
        "        return index_name_.lower() in lowered\n"
        "    return \"hnsw\" in lowered or \"index searches\" in lowered",
        caption="Listing 23 — Detecting the access path from the query plan.",
    )
    p(
        doc,
        "The result is stored on the experiment row and displayed in the user "
        "interface. The batch runner additionally prints a warning when a HNSW "
        "run reports used_index_scan = False, so a sequential-scan fallback "
        "cannot be mistaken for a fast approximate result. In the reported "
        "sweep this check passed on every HNSW row, and every exact row reported "
        "a sequential scan.",
    )

    h2(doc, "4.6 Classification and Metrics")
    p(doc, "app/services/classification.py", bold_lead="")
    table(
        doc,
        ["Function", "Signature", "Purpose"],
        [
            [
                "majority_vote",
                "(labels, distances=None, weighted=False) -> int",
                "The KNN decision rule. With weighted=True each neighbour "
                "contributes 1/(distance + ε). Ties resolve to the lowest "
                "label.",
            ],
            [
                "evaluate_predictions",
                "(y_true, y_pred, class_labels) -> ClassificationMetrics",
                "Accuracy, macro precision / recall / F1, the confusion matrix "
                "and the per-class breakdown.",
            ],
            [
                "recall_at_k",
                "(retrieved, ground_truth) -> float",
                "Conventional Recall@K: the mean fraction of the deterministic "
                "exact top-k identifiers that were retrieved.",
            ],
            [
                "tie_aware_recall_at_k",
                "(retrieved, candidates) -> float",
                "Tie-aware Recall@K: the fraction of retrieved ids that fall "
                "inside the K-th distance boundary, capped at 1.0.",
            ],
            [
                "repetition_summary",
                "(values) -> dict",
                "Median, standard deviation, min, max and a t-based 95% "
                "interval across repetitions.",
            ],
            [
                "latency_stats",
                "(latencies_ms) -> dict",
                "Average, median, p95, p99, minimum and maximum over every "
                "stored observation.",
            ],
        ],
        widths=[4.4, 6.2, 4.9],
    )
    p(
        doc,
        "evaluate_predictions treats its inputs as positional indices, "
        "0..n-1, and uses class_labels only for display. The confusion matrix "
        "is built directly from the positional indices, and each row of the "
        "per-class table reports the original label value.",
    )
    p(
        doc,
        "recall_at_k compares identifier sets, not values. It converts each "
        "retrieved list to a set, counts the members that appear in the "
        "deterministic ground truth, and averages the fraction across queries. "
        "tie_aware_recall_at_k does the same against the boundary candidate "
        "set and divides by k, so a valid but differently identified neighbour "
        "is not penalised. Both are reported and neither replaces the other.",
    )

    h2(doc, "4.7 Experiment Orchestration")
    p(doc, "app/services/experiments.py", bold_lead="")
    p(
        doc,
        "This module turns a configuration into a persisted, comparable "
        "result. It is where the two retrieval strategies are placed on the "
        "same footing.",
    )
    h3(doc, "4.7.1 ExperimentConfig")
    table(
        doc,
        ["Field", "Default", "Purpose"],
        [
            ["dataset_id", "—", "The dataset to query"],
            ["k", "5", "Neighbourhood size"],
            ["distance_metric", "L2", "The metric"],
            ["method", "'exact'", "'exact' or 'hnsw'"],
            ["ef_search", "None", "The search-effort parameter"],
            ["hnsw_m / hnsw_ef_construction", "16 / 64", "Build parameters"],
            ["max_queries", "2,000", "How many distinct test vectors to query"],
            ["repetitions", "1", "Complete timed passes over that query set"],
            ["seed", "0", "Seed recorded for reproducibility"],
            ["tie_tolerance", "1e-6 / 1e-6", "Relative and absolute tie tolerance"],
            ["query_sampling", "'first'", "'first' or 'stratified' query selection"],
            ["query_sample_seed", "0", "Seed for within-class query sampling"],
            ["benchmark_round / position", "None", "Balanced-round metadata from the runner"],
            ["benchmark_seed / order", "None", "Seed and full order of that round"],
            ["name", "None", "A label for the run"],
            ["weighted_voting", "False", "Distance-weighted voting"],
        ],
        widths=[5.2, 2.6, 7.7],
    )
    h3(doc, "4.7.2 run_experiment() — The Sequence")
    p(doc, "The function executes the following steps, in order:")
    numbered(
        doc,
        "Load the dataset row, deserialise its preprocessing state and compute "
        "its fingerprint.",
    )
    numbered(
        doc,
        "Select test ids with the configured mode and seed, load those vectors "
        "and persist the ids and both class distributions.",
    )
    numbered(
        doc,
        "Prepare the index: build or reuse the HNSW index for an approximate "
        "run, or refresh the planner statistics for an exact run.",
    )
    numbered(
        doc,
        "Compute the exact ground-truth ids and the boundary candidate set for "
        "every query.",
    )
    numbered(
        doc,
        "Execute the method under test for repetitions complete passes over "
        "the whole query set, in the same order, after one discarded warm-up "
        "query. Every latency is recorded with its repetition number.",
    )
    numbered(
        doc,
        "Compute conventional and tie-aware Recall@K per repetition, from that "
        "pass's neighbour sets.",
    )
    numbered(
        doc,
        "Classify both neighbour sets with the same majority-vote function and "
        "compute the classification metrics per repetition.",
    )
    numbered(
        doc,
        "Compute the latency summary over every stored observation, and the "
        "cross-repetition mean and standard deviation of the quality metrics.",
    )
    numbered(
        doc,
        "Persist the experiment row, one experiment_repetitions row per pass "
        "and one query_latencies row per measured query, then write the "
        "non-overlapping timing fields back to the experiment row.",
    )
    p(
        doc,
        "The exact query orders by (distance, id) so repeated exact retrieval is "
        "deterministic when distances tie. The HNSW query deliberately keeps "
        "only the index-compatible distance ordering; adding an id sort would "
        "make the planner abandon the index.",
    )
    p(
        doc,
        "Step 4 deserves emphasis: the ground truth is recomputed for every "
        "run, using the exact query, in the same session and against the same "
        "data as the measured run. Recall@K is therefore always a "
        "like-for-like comparison.",
    )
    p(
        doc,
        "Step 7 is what makes the accuracy comparison valid. Both neighbour "
        "sets are passed to the same voting function, so the only difference "
        "between the two arms of the experiment is the retrieval.",
    )
    h3(doc, "4.7.3 Comparison")
    p(
        doc,
        "compare_experiments(exact_id, approximate_id) fetches both rows and "
        "returns them with a deltas block containing the difference in "
        "accuracy, macro F1, Recall@K and the three latency percentiles, plus "
        "the speed-up ratio computed as the exact average latency divided by "
        "the approximate average latency.",
    )

    h2(doc, "4.8 Dataset Acquisition")
    p(doc, "app/services/datasets.py", bold_lead="")
    table(
        doc,
        ["Function", "Purpose"],
        [
            ["covertype_cache_paths()", "The paths of the cached archive and the extracted CSV"],
            [
                "download_covertype(force=False)",
                "Fetches the archive from UCI and extracts it, or uses the "
                "cache. The extracted file is kept gzipped.",
            ],
            [
                "load_covertype(max_rows=None)",
                "Reads the cached CSV into a DataFrame with the 55 UCI column "
                "names, optionally subsampling.",
            ],
            [
                "load_uploaded_csv(raw, target_column)",
                "Parses an uploaded CSV payload, falling back to latin-1 when "
                "the UTF-8 decode fails.",
            ],
            [
                "stratified_split(df, target_column, test_size, train_fraction, seed)",
                "A stratified train/test split that preserves the class "
                "distribution.",
            ],
            [
                "sample_training_rows(df, target_column, max_rows, seed)",
                "A stratified subsample of the training pool, used for the "
                "dataset-size experiments.",
            ],
            ["class_distribution(labels)", "The per-class counts, for display"],
        ],
        widths=[6.4, 9.1],
    )
    p(
        doc,
        "The Covertype column list has 55 entries: 10 continuous features, 4 "
        "wilderness one-hot features, 40 soil one-hot features and the label. "
        "An early version of this list included derived 'alternative' columns "
        "that are not present in the UCI file, which caused the loader to "
        "produce a 62-column frame with an empty label column.",
    )
    note(
        doc,
        "sample_training_rows samples within each class rather than uniformly, "
        "so the class proportions of the training pool are preserved at every "
        "size. A uniform subsample would quietly change the class balance and "
        "make accuracy non-comparable across sizes.",
    )

    h2(doc, "4.9 The HTTP API")
    h3(doc, "4.9.1 Dataset endpoints")
    table(
        doc,
        ["Method", "Path", "Purpose"],
        [
            ["GET", "/api/datasets", "List datasets"],
            ["GET", "/api/datasets/{id}", "One dataset"],
            ["DELETE", "/api/datasets/{id}", "Drop the dataset and its tables"],
            ["POST", "/api/datasets/covertype", "Ingest the seeded dataset"],
            ["POST", "/api/datasets/upload", "Ingest an uploaded CSV"],
            ["GET", "/api/datasets/covertype/status", "Report the cache state"],
        ],
        widths=[2.2, 6.2, 7.1],
    )
    p(
        doc,
        "_ingest_frame is the shared pipeline behind both ingestion routes. It "
        "splits, samples, fits the preprocessor on the training split, "
        "transforms both splits, creates the vector tables and bulk-loads "
        "them. If the load fails, the partially-created dataset row and its "
        "tables are removed, so a failed ingestion leaves no residue.",
    )
    h3(doc, "4.9.2 Experiment endpoints")
    table(
        doc,
        ["Method", "Path", "Purpose"],
        [
            ["GET", "/api/experiments", "List experiments, optionally by dataset"],
            ["GET", "/api/experiments/{id}", "One experiment"],
            ["DELETE", "/api/experiments/{id}", "Delete an experiment"],
            ["POST", "/api/experiments/run", "Run an experiment"],
            ["GET", "/api/experiments/compare", "Compare an exact and an HNSW run"],
            ["GET", "/api/experiments/{id}/latencies", "Per-query latency samples"],
            ["GET", "/api/experiments/{id}/repetitions", "Per-repetition results"],
            ["GET", "/api/experiments/{id}/explain", "Configuration and access path"],
        ],
        widths=[2.2, 6.2, 7.1],
    )
    note(
        doc,
        "Route order matters. The static paths /run and /compare are declared "
        "before the parameterised /{experiment_id}, because FastAPI matches "
        "routes in declaration order and the parameterised route will "
        "otherwise swallow the static one — producing the confusing error "
        "“Input should be a valid integer, unable to parse string as an "
        "integer, input: 'compare'”.",
        label="Trap",
    )
    h3(doc, "4.9.3 Health endpoint")
    p(
        doc,
        "GET /api/health returns the service status, the database connection "
        "state, the installed pgvector version and the counts of datasets and "
        "experiments. The frontend polls it to display a live status badge.",
    )

    h2(doc, "4.10 The Batch Experiment Runner")
    p(doc, "scripts/run_experiments.py", bold_lead="")
    p(
        doc,
        "This script runs the full experimental matrix without the HTTP "
        "layer. It is the tool used to produce the results in the thesis.",
    )
    code(
        doc,
        "python -m scripts.run_experiments \\\n"
        "    --dataset-id 116 \\\n"
        "    --sizes 10000,50000 \\\n"
        "    --ef-search 10,25,50,100,200,400 \\\n"
        "    --k 5 --metric l2 --max-queries 1000 \\\n"
        "    --repetitions 3 --seed 0 \\\n"
        "    --rounds 3 --order-seed 0 \\\n"
        "    --query-sampling stratified --query-sample-seed 0 \\\n"
        "    --output ..\\docs\\results\\sweep.csv \\\n"
        "    --json   ..\\docs\\results\\sweep.json \\\n"
        "    --repetition-details ..\\docs\\results\\repetition_details.csv \\\n"
        "    --repetition-summary ..\\docs\\results\\repetition_summary.csv \\\n"
        "    --query-samples ..\\docs\\results\\query_samples.json",
        caption="Listing 24 — Running the experiment matrix.",
    )
    p(
        doc,
        "Because the dataset-size experiment mutates the training pool, the "
        "script snapshots the full pool before the sweep and restores it "
        "afterwards, in a finally block. The snapshot is written back with "
        "COPY, which preserves the row identifiers and therefore keeps the "
        "Recall@K references consistent across the sweep.",
    )
    p(
        doc,
        "The runner enforces six guarantees that are easy to lose when results "
        "are produced by hand. It writes 64 columns, separating unique queries, "
        "repetitions, rounds and stored observations. It selects queries by a "
        "seeded stratified sample, persists their ids and exports them to "
        "query_samples.json. It executes the matrix in seeded balanced rounds "
        "so every configuration runs early and late, without rebuilding the "
        "index between ef_search values. It reads the per-repetition rows back "
        "from the database before writing the repetition artefacts. It prints a "
        "warning when a HNSW row reports used_index_scan = False. And it records "
        "index_reused together with a NULL index_build_ms on reuse. The v1 "
        "artifacts remain under docs/results/v1 for comparison.",
    )

    h2(doc, "4.11 The Test Suite")
    p(doc, "backend/tests/", bold_lead="")
    table(
        doc,
        ["File", "Covers"],
        [
            [
                "test_classification.py",
                "Majority voting, weighted voting, the confusion matrix, the "
                "macro metrics, Recall@K and the latency percentiles",
            ],
            [
                "test_tie_aware_recall.py",
                "Conventional versus tie-aware Recall@K: tied boundaries, "
                "candidate sets, the cap, and identical results on untied data",
            ],
            [
                "test_dataset_stats.py",
                "Dataset fingerprinting, duplicate groups, mixed-label groups "
                "and boundary-tie measurement",
            ],
            [
                "test_preprocessing.py",
                "Column-type inference, category mapping, the scaling "
                "contract, unseen categories and the leakage contract",
            ],
            [
                "test_datasets.py",
                "Stratification, class-proportion preservation, the absence of "
                "row leakage between splits, and the Covertype column layout",
            ],
            [
                "test_vector_store.py",
                "Table creation, exact retrieval, index building and reuse, "
                "plan verification, ground-truth determinism and repetition "
                "accounting",
            ],
            [
                "test_benchmark_order.py",
                "Seeded balanced rounds: one run per configuration per round, "
                "reproducibility, seed variation and early/late coverage",
            ],
            [
                "test_query_sampling.py",
                "Stratified query ids, seed reproducibility, rare classes, "
                "class proportions, no train/test overlap and persistence",
            ],
            [
                "test_timing_accounting.py",
                "Non-overlapping phase intervals, retrieval totals matching "
                "stored samples, fresh build versus reuse",
            ],
            [
                "test_exact_tie_breaking.py",
                "Deterministic exact ids on tied data, HNSW index preservation "
                "and tie-aware recall credit",
            ],
            [
                "test_tie_tolerance_metrics.py",
                "L2, cosine and L1 tolerance cases: exact ties, near/far "
                "distances, tiny and large boundaries, >K candidates",
            ],
            [
                "test_run_experiments.py",
                "The runner's field set, CSV/JSON agreement, persisted latency "
                "count, balanced rounds and a full main() smoke test",
            ],
        ],
        widths=[5.0, 10.5],
    )
    p(
        doc,
        "The suite contains 211 passing tests and one documented skip (cosine "
        "distance is discrete for collinear vectors, so a near-zero non-zero "
        "boundary cannot be constructed). It runs against a real PostgreSQL + "
        "pgvector instance; nothing is mocked, because a mocked vector store "
        "would not prove that pgvector's planner behaves as assumed. The "
        "integration tests are skipped automatically when PostgreSQL is "
        "unreachable, so the unit tests still run on a machine without a "
        "database.",
    )


# ---------------------------------------------------------------------------
# Part 5
# ---------------------------------------------------------------------------
def build_frontend(doc) -> None:
    h1(doc, "5. The Frontend")

    p(
        doc,
        "The frontend is a Next.js 16 application on the App Router. It is "
        "entirely a client-side data application: the server renders the page "
        "shell and the locale, and everything that depends on experimental "
        "results is fetched from the backend by the browser.",
    )

    h2(doc, "5.1 Configuration Files")
    h3(doc, "5.1.1 next.config.ts")
    p(
        doc,
        "The configuration is deliberately small. output is set to standalone "
        "so the app can be deployed as a self-contained server, and the "
        "next-intl plugin is wired to the request configuration. The "
        "turbopack.root option pins the build root to the project directory, "
        "which suppresses a warning about a stray lockfile in a parent "
        "directory.",
    )
    code(
        doc,
        "const withNextIntl = createNextIntlPlugin('./i18n/request.ts')\n"
        "\n"
        "const nextConfig: NextConfig = {\n"
        "  output: 'standalone',\n"
        "  turbopack: {\n"
        "    root: process.cwd(),\n"
        "  },\n"
        "  experimental: {\n"
        "    serverActions: {\n"
        "      bodySizeLimit: '256mb',\n"
        "    },\n"
        "  },",
        caption="Listing 25 — The Next.js configuration.",
    )
    h3(doc, "5.1.2 proxy.ts")
    p(
        doc,
        "Next.js 16 renamed the middleware convention to proxy. This file "
        "delegates to the next-intl middleware, which is what adds the locale "
        "prefix to incoming URLs.",
    )
    code(
        doc,
        "import createMiddleware from 'next-intl/middleware'\n"
        "import { NextRequest } from 'next/server'\n"
        "import { routing } from './i18n/routing'\n"
        "\n"
        "const handleI18nRouting = createMiddleware(routing)\n"
        "\n"
        "export default function proxy(request: NextRequest) {\n"
        "  return handleI18nRouting(request)\n"
        "}\n"
        "\n"
        "export const config = {\n"
        "  matcher: ['/((?!api|_next|_vercel|.*\\\\..*).*)'],\n"
        "}",
        caption="Listing 26 — The proxy.",
    )
    h3(doc, "5.1.3 eslint.config.mjs")
    p(
        doc,
        "eslint-config-next version 16 is a native flat configuration, so the "
        "config is a direct composition rather than the FlatCompat shim used "
        "in older projects.",
    )
    code(
        doc,
        "import { defineConfig } from 'eslint/config'\n"
        "import nextVitals from 'eslint-config-next/core-web-vitals'\n"
        "import nextTypeScript from 'eslint-config-next/typescript'\n"
        "\n"
        "export default defineConfig([\n"
        "  {\n"
        "    ignores: [\n"
        "      'node_modules/**',\n"
        "      '.next/**',\n"
        "      'out/**',\n"
        "      'build/**',\n"
        "      'next-env.d.ts',\n"
        "    ],\n"
        "  },\n"
        "  ...nextVitals,\n"
        "  ...nextTypeScript,\n"
        "])",
        caption="Listing 27 — The lint configuration.",
    )

    h2(doc, "5.2 Internationalisation")
    h3(doc, "5.2.1 routing.ts")
    p(
        doc,
        "The routing configuration declares the two supported locales, the "
        "default locale and disables automatic locale detection so that the "
        "URL always carries an explicit prefix. This makes every page "
        "addressable in both languages.",
    )
    code(
        doc,
        "export const routing = defineRouting({\n"
        "  locales: ['en', 'fa'],\n"
        "  defaultLocale: 'en',\n"
        "  localePrefix: 'always',\n"
        "  localeDetection: false,\n"
        "})",
        caption="Listing 28 — The locale routing configuration.",
    )
    h3(doc, "5.2.2 request.ts")
    p(
        doc,
        "The request configuration resolves the locale and loads the "
        "corresponding message catalogue. The import is dynamic and keyed by "
        "locale, so only the active catalogue is bundled into the page.",
    )
    h3(doc, "5.2.3 navigation.ts")
    p(
        doc,
        "The navigation helpers wrap Next.js's own navigation APIs so that "
        "they are locale-aware. The Link component automatically prefixes the "
        "current locale, and the router's replace method accepts a locale "
        "option, which is how the language switcher works without a page "
        "reload.",
    )
    code(
        doc,
        "const switchLocale = (next: string) => {\n"
        "  startTransition(() => {\n"
        "    router.replace(pathname, { locale: next })\n"
        "  })\n"
        "}",
        caption="Listing 29 — Switching the locale without a full page load (components/app-header.tsx).",
    )
    note(
        doc,
        "An earlier version used window.location.href to switch the locale. "
        "That triggers the @next/next/no-location-assign-relative-destination "
        "lint rule and forces a full reload. The router.replace approach is "
        "both cleaner and keeps the React state alive.",
        label="Note",
    )
    h3(doc, "5.2.4 messages/en.json and messages/fa.json")
    p(
        doc,
        "The message catalogues are organised by namespace: common, health, "
        "overview, datasets, experiments, compare and methodology. Every key "
        "used by a component is declared in both files, which keeps the two "
        "languages in step.",
    )

    h2(doc, "5.3 The Root Layout")
    p(doc, "app/[locale]/layout.tsx", bold_lead="")
    table(
        doc,
        ["Element", "Purpose"],
        [
            ["generateStaticParams", "Pre-renders both locales at build time"],
            ["generateMetadata", "Locale-aware title and description"],
            ["localFont for Inter", "The Latin typeface, exposed as --font-sans-ui"],
            ["localFont for Vazirmatn", "The Persian typeface, exposed as --font-fa-ui"],
            ["NextIntlClientProvider", "Makes the messages available to client components"],
            ["QueryProviders", "The TanStack Query client"],
            ["ThemeProvider", "Light and dark theme support"],
            ["html lang / dir", "Sets the language and the writing direction"],
        ],
        widths=[5.4, 10.1],
    )
    code(
        doc,
        "<html\n"
        "  lang={locale}\n"
        "  dir={isRtl ? 'rtl' : 'ltr'}\n"
        "  suppressHydrationWarning\n"
        "  className={`${interFont.variable} ${vazirFont.variable}`}\n"
        ">",
        caption="Listing 30 — The document element with the RTL switch.",
    )
    p(
        doc,
        "The stylesheet applies the Persian typeface only under the fa "
        "locale, using an attribute selector on the html element, so the "
        "English pages keep the Latin face.",
    )

    h2(doc, "5.4 The Data Layer")
    h3(doc, "5.4.1 types/api.ts")
    p(
        doc,
        "A hand-written mirror of the backend's Pydantic schemas. It defines "
        "DatasetSummary, IngestResponse, Experiment, ExperimentRequest, "
        "ExperimentComparison, LatencySamples and HealthStatus, together with "
        "the DistanceMetric and RetrievalMethod unions.",
    )
    h3(doc, "5.4.2 lib/api.ts — The API Client")
    p(
        doc,
        "A single typed wrapper around fetch. Every function corresponds to "
        "one backend endpoint, and the module is the only place in the "
        "frontend that knows the backend's URL structure.",
    )
    table(
        doc,
        ["Function", "Endpoint", "Returns"],
        [
            ["getHealth()", "GET /health", "HealthStatus"],
            ["listDatasets()", "GET /datasets", "DatasetSummary[]"],
            ["getDataset(id)", "GET /datasets/{id}", "DatasetSummary"],
            ["deleteDataset(id)", "DELETE /datasets/{id}", "{ deleted }"],
            ["ingestCovertype(body)", "POST /datasets/covertype", "IngestResponse"],
            ["uploadDataset(form)", "POST /datasets/upload", "IngestResponse"],
            ["listExperiments(datasetId?)", "GET /experiments", "Experiment[]"],
            ["getExperiment(id)", "GET /experiments/{id}", "Experiment"],
            ["deleteExperiment(id)", "DELETE /experiments/{id}", "{ deleted }"],
            ["runExperiment(body)", "POST /experiments/run", "Experiment"],
            ["compareExperiments(a, b)", "GET /experiments/compare", "ExperimentComparison"],
            ["getLatencySamples(id)", "GET /experiments/{id}/latencies", "LatencySamples"],
        ],
        widths=[5.0, 6.0, 4.5],
    )
    code(
        doc,
        "export class ApiError extends Error {\n"
        "  constructor(\n"
        "    message: string,\n"
        "    readonly status: number,\n"
        "  ) {\n"
        "    super(message)\n"
        "    this.name = 'ApiError'\n"
        "  }\n"
        "}\n"
        "\n"
        "async function request<T>(path: string, init?: RequestInit): Promise<T> {\n"
        "  const response = await fetch(`${API_BASE_URL}${API_PREFIX}${path}`, {\n"
        "    ...init,\n"
        "    headers: {\n"
        "      'Content-Type': 'application/json',\n"
        "      ...init?.headers,\n"
        "    },\n"
        "    cache: 'no-store',\n"
        "  })\n"
        "\n"
        "  if (!response.ok) {\n"
        "    let detail = response.statusText\n"
        "    try {\n"
        "      const payload = await response.json()\n"
        "      detail = payload.detail ?? JSON.stringify(payload)\n"
        "    } catch {\n"
        "      /* response body was not JSON */\n"
        "    }\n"
        "    throw new ApiError(detail, response.status)\n"
        "  }\n"
        "\n"
        "  if (response.status === 204) return undefined as T\n"
        "  return (await response.json()) as T\n"
        "}",
        caption="Listing 31 — The API client core.",
    )
    p(
        doc,
        "The error handling extracts the backend's detail field, which is "
        "where FastAPI places validation messages and the exception text "
        "raised by the services. The upload function is separate because it "
        "sends multipart form data rather than JSON.",
    )
    h3(doc, "5.4.3 lib/utils.ts — Formatting Helpers")
    table(
        doc,
        ["Function", "Purpose"],
        [
            ["cn(...inputs)", "The Tailwind class-merging helper used by every component"],
            ["formatMs(ms, digits)", "Formats a duration, switching to seconds above 1000 ms"],
            ["formatPercent(value, digits)", "Formats a 0..1 ratio as a percentage"],
            ["formatDelta(value, digits)", "Formats a signed difference with an explicit sign"],
            ["formatNumber(value)", "Formats an integer with thousands separators"],
        ],
        widths=[5.4, 10.1],
    )
    h3(doc, "5.4.4 providers/")
    p(
        doc,
        "query-provider.tsx creates the TanStack Query client with a stale "
        "time of thirty seconds and disables refetching on window focus. "
        "theme-providers.tsx is a thin wrapper around next-themes that "
        "enables the class-based dark mode.",
    )

    h2(doc, "5.5 The Components")
    h3(doc, "5.5.1 app-shell.tsx and app-header.tsx")
    p(
        doc,
        "The shell provides the page frame, the header and the footer. The "
        "header carries the product mark, the primary navigation, a theme "
        "toggle and the language switcher. The navigation items are defined "
        "once as a constant and rendered in a loop, and the active item is "
        "determined by comparing the current pathname.",
    )
    h3(doc, "5.5.2 health-badge.tsx")
    p(
        doc,
        "Polls the health endpoint every thirty seconds and renders a status "
        "pill, the pgvector version and the dataset and experiment counts. "
        "The pill uses a pulsing dot to signal a live connection.",
    )
    h3(doc, "5.5.3 overview-content.tsx")
    p(
        doc,
        "The overview presents the research question, five headline metric "
        "cards, a numbered quick-start guide and a list of the most recent "
        "experiments. The metric cards pull the exact and the approximate run "
        "from the experiment list and compute the speed-up ratio between them.",
    )
    h3(doc, "5.5.4 datasets-content.tsx")
    p(
        doc,
        "Two tabs: one to ingest the seeded Covertype dataset with "
        "configurable sampling limits, and one to upload a CSV. Below them, a "
        "table lists every dataset with its size, dimensionality and a delete "
        "action. Both mutations invalidate the datasets query on success and "
        "report the outcome with a toast.",
    )
    h3(doc, "5.5.5 experiment-runner.tsx — The Control Surface")
    p(
        doc,
        "This component owns every experimental parameter. It is a client "
        "component with local state for each parameter and a mutation that "
        "posts the configuration to the backend.",
    )
    table(
        doc,
        ["Parameter", "Control", "Backend field"],
        [
            ["Dataset", "Select", "dataset_id"],
            ["k", "Number input", "k"],
            ["Metric", "Select (L2 / cosine / L1)", "distance_metric"],
            ["Method", "Tabs (exact / HNSW)", "method"],
            ["ef_search", "Slider from 5 to 800", "ef_search"],
            ["hnsw.m", "Number input", "hnsw_m"],
            ["ef_construction", "Number input", "hnsw_ef_construction"],
            ["Query count", "Number input", "max_queries"],
            ["Weighted voting", "Checkbox", "weighted_voting"],
        ],
        widths=[4.0, 6.4, 5.1],
    )
    p(
        doc,
        "The HNSW-specific controls are only rendered when the HNSW method is "
        "selected, and the corresponding fields are sent as null for the "
        "exact method. The mutation invalidates the experiment list and the "
        "selected experiment's cache on success.",
    )
    h3(doc, "5.5.6 experiment-result-panel.tsx")
    p(
        doc,
        "Displays the metrics of the selected experiment as six stat tiles, "
        "then the confusion matrix and the per-class table side by side, then "
        "the latency distribution. The confusion matrix cells are shaded "
        "proportionally to the column total, which makes the dominant "
        "confusions visible at a glance.",
    )
    p(
        doc,
        "The latency histogram is drawn from the raw samples returned by the "
        "backend: the values are binned into twenty-four buckets and "
        "rendered as a flex row of bars whose heights are proportional to the "
        "bucket counts. The percentile figures are computed by the backend "
        "and displayed alongside.",
    )
    h3(doc, "5.5.7 experiments-content.tsx")
    p(
        doc,
        "Composes the runner, the result panel and the experiment history "
        "table. The history table lists every run with its key metrics and "
        "lets the user select one to inspect it.",
    )
    h3(doc, "5.5.8 compare-content.tsx — The Comparison View")
    p(
        doc,
        "Two selects choose the exact run and the approximate run. A "
        "comparison table then shows each metric for both runs with the "
        "difference, coloured green or red depending on whether the change "
        "is favourable. Three conclusion tiles summarise the recall impact, "
        "the accuracy impact and the speed-up.",
    )
    p(
        doc,
        "Below the table, two Recharts visualisations summarise the whole "
        "sweep across the dataset. The first plots Recall@K and the median "
        "latency against ef_search on twin axes — the trade-off curve. The "
        "second plots accuracy against ef_search as a bar chart — the cost of "
        "the approximation. The data for both is derived from every HNSW run "
        "of the dataset, so they update as experiments are added.",
    )
    h3(doc, "5.5.9 methodology-content.tsx")
    p(
        doc,
        "A static page that documents the architecture, the six-step "
        "pipeline, the measured metrics and the three methodological notes "
        "that affect the validity of the results.",
    )

    h2(doc, "5.6 The Pages")
    table(
        doc,
        ["Route", "Component", "Responsibility"],
        [
            ["/[locale]", "app/[locale]/page.tsx", "Overview"],
            ["/[locale]/datasets", "app/[locale]/datasets/page.tsx", "Dataset ingestion"],
            ["/[locale]/experiments", "app/[locale]/experiments/page.tsx", "Experiment runner and results"],
            ["/[locale]/compare", "app/[locale]/compare/page.tsx", "Side-by-side comparison"],
            ["/[locale]/methodology", "app/[locale]/methodology/page.tsx", "Method documentation"],
        ],
        widths=[4.2, 6.0, 5.3],
    )
    p(
        doc,
        "Every page follows the same pattern: a server component that awaits "
        "the params, calls setRequestLocale, loads the translations with "
        "getTranslations and renders the AppShell around a client component "
        "that does the interactive work. This keeps the data fetching on the "
        "client, where TanStack Query can cache and revalidate it, while the "
        "page shell is still statically rendered.",
    )
    note(
        doc,
        "A page must not call useSearchParams directly. Doing so opts the "
        "route into a client-side rendering bailout and the production build "
        "fails with “useSearchParams() should be wrapped in a suspense "
        "boundary”. The header avoids it by reading window.location.search "
        "directly when needed, or by using the locale-aware router.",
        label="Trap",
    )


# ---------------------------------------------------------------------------
# Part 6
# ---------------------------------------------------------------------------
def build_appendix(doc) -> None:
    h1(doc, "6. Appendix")

    h2(doc, "6.1 End-to-End Data Flow")
    code(
        doc,
        "CSV / Covertype\n"
        "      │\n"
        "      ▼\n"
        "stratified_split ──────────────► train_df            test_df\n"
        "      │                              │                   │\n"
        "      │                    fit_preprocessor(train_df)    │  ← statistics from\n"
        "      │                              │                   │    training only\n"
        "      │                              ▼                   │\n"
        "      │              standardise 10 continuous,          │\n"
        "      │              keep 44 indicators 0/1              │\n"
        "      │              (one-hot nominal for uploads)       │\n"
        "      │                              │                   │\n"
        "      │                              ▼                   ▼\n"
        "      │                    transform(train)      transform(test)\n"
        "      │                              │                   │\n"
        "      ▼                              ▼                   ▼\n"
        "  datasets row            COPY → fv_<id>       COPY → tv_<id>\n"
        "                                   │\n"
        "                    ┌──────────────┴───────────────┐\n"
        "                    ▼                              ▼\n"
        "           exact (function form)          HNSW (operator form)\n"
        "           ground truth ids +             N repetitions ×\n"
        "           boundary candidates             every query\n"
        "                    │                              │\n"
        "                    └──────────────┬───────────────┘\n"
        "                                   ▼\n"
        "          majority vote → metrics per repetition → experiments\n"
        "                    │\n"
        "                    ▼\n"
        "     experiment_repetitions + query_latencies → CSV artefacts",
        caption="Listing 32 — The complete data flow of one experiment (diagram).",
    )

    h2(doc, "6.2 Extension Recipes")
    h3(doc, "6.2.1 Add a new distance metric")
    numbered(
        doc,
        "Add a member to the DistanceMetric enum in "
        "app/services/metrics.py, giving it its operator, its function and "
        "its label.",
    )
    numbered(
        doc,
        "Add the matching operator class to the ops dictionary in "
        "_build_hnsw_index.",
    )
    numbered(
        doc,
        "Add the value to the Literal union in ExperimentRequest and to the "
        "Select in the frontend's experiment runner.",
    )
    numbered(
        doc,
        "Verify that pgvector supports the metric for HNSW indexing — not "
        "every operator has an indexable operator class.",
    )
    h3(doc, "6.2.2 Add a new metric")
    numbered(
        doc,
        "Compute it in run_experiment() in app/services/experiments.py.",
    )
    numbered(
        doc,
        "Add a column to the experiments table in app/db/schema.py, and a "
        "key to the payload dictionary.",
    )
    numbered(
        doc,
        "Add the field to the Experiment type in frontend/types/api.ts and "
        "display it in the result panel.",
    )
    h3(doc, "6.2.3 Add a new page")
    numbered(
        doc,
        "Create app/[locale]/<page>/page.tsx as a server component that "
        "awaits params, calls setRequestLocale and renders the AppShell.",
    )
    numbered(
        doc,
        "Create the interactive content as a separate client component "
        "under components/.",
    )
    numbered(
        doc,
        "Add the navigation entry to NAV_ITEMS in components/app-header.tsx "
        "and the message keys to both message catalogues.",
    )
    h3(doc, "6.2.4 Support a new dataset source")
    numbered(
        doc,
        "Add a loader to app/services/datasets.py returning a DataFrame with "
        "a target column.",
    )
    numbered(
        doc,
        "Add an ingestion endpoint that calls _ingest_frame with the new "
        "source name.",
    )

    h2(doc, "6.3 Troubleshooting")
    table(
        doc,
        ["Symptom", "Cause", "Fix"],
        [
            [
                "“extension \\\"vector\\\" is not available”",
                "extension_control_path does not point at a directory "
                "containing an extension sub-directory",
                "Put the .control and .sql files in <path>/extension/ and "
                "reload the configuration",
            ],
            [
                "“could not access file \\\"vector\\\"”",
                "The PostgreSQL service account cannot read the files",
                "icacls the pgvector directory granting NETWORK SERVICE read "
                "access",
            ],
            [
                "“column does not have dimensions”",
                "The HNSW index was created over an unconstrained vector column",
                "Declare the column as vector(dim) in the per-dataset table",
            ],
            [
                "Recall@K is identical for every ef_search value",
                "SET LOCAL was used in autocommit mode and silently discarded",
                "Use a session-level SET with an explicit RESET",
            ],
            [
                "The HNSW run reports the same latency as the exact run",
                "The planner reverted to a sequential scan, or the session "
                "ef_search was not applied",
                "Inspect the access path and the applied ef_search with the "
                "explain endpoint; a HNSW row without an index scan is not an "
                "HNSW measurement",
            ],
            [
                "Recall@K falls when ef_search rises",
                "Not necessarily a bug. HNSW traversal is heuristic, so a "
                "higher effort can change which candidate list survives; the "
                "curve is not guaranteed to be monotonic",
                "Re-run before concluding; treat an inversion as a result to "
                "explain, not as an automatic failure",
            ],
            [
                "The compare endpoint returns a 422",
                "A static route was shadowed by a parameterised route",
                "Declare /run and /compare before /{experiment_id}",
            ],
            [
                "The production build fails with a useSearchParams error",
                "A component calls useSearchParams outside a Suspense boundary",
                "Avoid it, or wrap the component in Suspense",
            ],
            [
                "“extra data after last expected column” during COPY",
                "FORMAT csv was used with a vector literal containing commas",
                "Use the default text format for COPY",
            ],
            [
                "A psycopg error mentions Identifier('…')",
                "A sql.Identifier was interpolated into a plain Python string",
                "Use sql.SQL(...).format(...) instead of str.format",
            ],
        ],
        widths=[4.6, 5.4, 5.5],
    )

    h2(doc, "6.4 Command Reference")
    table(
        doc,
        ["Purpose", "Command"],
        [
            ["Start the backend", "cd backend && .venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000"],
            ["Start the frontend", "cd frontend && bun run dev"],
            ["Run the tests", "cd backend && .venv\\Scripts\\python -m pytest"],
            ["Lint the frontend", "cd frontend && bun run lint"],
            ["Build the frontend", "cd frontend && bun run build"],
            ["Type-check", "cd frontend && bun run typecheck"],
            ["Run the experiment matrix", "cd backend && .venv\\Scripts\\python -m scripts.run_experiments --dataset-id 116 --sizes 10000,50000 --ef-search 10,25,50,100,200,400 --repetitions 3"],
            ["Measure duplicates and ties", "cd backend && .venv\\Scripts\\python -m scripts.dataset_statistics --dataset-id 116"],
            ["Regenerate this guide", "python docs/tools/build_guide.py"],
            ["Regenerate the thesis", "python docs/tools/build_thesis.py"],
            ["Verify the whole stack", "scripts\\verify.bat"],
        ],
        widths=[4.4, 11.1],
        font_size=8,
    )


# ---------------------------------------------------------------------------
def main() -> int:
    doc = new_document(body_font="Calibri", size=11)

    build_cover(doc)
    build_how_to_use(doc)

    doc.add_heading("Table of Contents", level=1)
    add_table_of_contents(doc)
    page_break(doc)

    build_orientation(doc)
    build_environment(doc)
    build_database(doc)
    build_backend(doc)
    build_frontend(doc)
    build_appendix(doc)

    add_page_number_footer(doc, "KNN Classification Platform — Project Guide")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"Guide written to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
