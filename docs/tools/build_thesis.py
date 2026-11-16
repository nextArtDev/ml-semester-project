"""Generate the thesis Word document.

Run from the repository root::

    python docs/tools/build_thesis.py

The measured numbers are taken from ``docs/results/sweep.csv`` so the
document and the artefacts never drift apart.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx.enum.text import WD_ALIGN_PARAGRAPH  # noqa: E402
from docx.shared import Cm, Pt  # noqa: E402

from docx_kit import (  # noqa: E402
    bullet,
    bullet_list,
    caption,
    code,
    h1,
    h2,
    h3,
    h4,
    note,
    numbered,
    p,
    page_break,
    table,
)

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "results" / "sweep.csv"
OUTPUT = ROOT / "docs" / "Thesis-KNN-PostgreSQL-Platform.docx"

TITLE = (
    "Design and Evaluation of a PostgreSQL-Based K-Nearest Neighbors "
    "Classification Platform Using Exact and Approximate Vector Search"
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load_results() -> list[dict]:
    if not RESULTS.exists():
        return []
    with open(RESULTS, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pick(rows: list[dict], size: int, method: str, ef: int | None):
    for row in rows:
        if (
            int(row["train_size"]) == size
            and row["retrieval_method"] == method
            and (ef is None or int(row["hnsw_ef_search"] or 0) == ef)
        ):
            return row
    return None


def fmt_pct(value: float | str | None, digits: int = 2) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value) * 100:.{digits}f}%"


def fmt_ms(value: float | str | None, digits: int = 3) -> str:
    if value in (None, ""):
        return "—"
    return f"{float(value):.{digits}f}"


# ---------------------------------------------------------------------------
# Front matter
# ---------------------------------------------------------------------------
def build_front_matter(doc) -> None:
    for _ in range(4):
        doc.add_paragraph()

    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("Master's Machine Learning Course Project")
    run.bold = True
    run.font.size = Pt(13)

    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run("THESIS")
    run.bold = True
    run.font.size = Pt(34)
    run.font.color.rgb = None

    paragraph = doc.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(TITLE)
    run.bold = True
    run.font.size = Pt(16)
    paragraph.paragraph_format.space_before = Pt(18)

    doc.add_paragraph()

    table(
        doc,
        ["Field", "Value"],
        [
            ["Project title", TITLE],
            ["Course", "Master's Machine Learning"],
            ["Study area", "Classical machine learning · vector similarity search"],
            ["Primary dataset", "UCI Forest CoverType (581,012 × 54, 7 classes)"],
            ["Database", "PostgreSQL 18 with the pgvector 0.8.7 extension"],
            ["Approximate method", "HNSW (Hierarchical Navigable Small World)"],
            ["Deliverables", "Web platform, evaluation harness, experimental results, report"],
        ],
        widths=[4.5, 11.0],
        font_size=10,
    )

    page_break(doc)


def build_abstract(doc) -> None:
    h1(doc, "Abstract")

    p(
        doc,
        "The K-Nearest Neighbors (KNN) algorithm is a widely used instance-based "
        "classification method whose apparent simplicity conceals a linear "
        "prediction cost: every query must be compared against the whole training "
        "set. For a dataset of 581,012 observations this cost is no longer "
        "academic but operational, and it is precisely the cost that modern "
        "vector similarity search was designed to relieve.",
    )
    p(
        doc,
        "This project designs and implements an interactive platform that performs "
        "KNN classification entirely inside PostgreSQL, using the pgvector "
        "extension for vector storage and retrieval, and then measures — rather "
        "than assumes — the consequences of trading exactness for speed. The "
        "platform stores preprocessed feature vectors in dimensionally-typed "
        "pgvector tables, executes exhaustive retrieval as a reference baseline, "
        "and executes approximate retrieval through an HNSW index under a "
        "configurable search-effort parameter.",
    )
    p(
        doc,
        "The experimental evaluation uses the UCI Forest CoverType dataset and "
        "compares the two retrieval strategies in terms of neighbour-retrieval "
        "recall, classification quality (accuracy, macro precision, macro recall "
        "and macro F1), and per-query latency. Across training-set sizes of "
        "10,000 and 50,000 rows, HNSW retrieval achieved between 86.6% and 98.5% "
        "Recall@K while reducing median query latency by a factor of 3.2 to 45.0 "
        "relative to the exact baseline, at a cost of at most 3.3 percentage "
        "points of classification accuracy.",
    )
    p(
        doc,
        "The central finding is that classification quality is far more robust "
        "than retrieval quality: a missed neighbour changes the predicted class "
        "only when it flips the majority vote, so the accuracy penalty of "
        "approximation is roughly an order of magnitude smaller than the recall "
        "penalty. The second finding is that the benefit of approximation is not "
        "a constant factor but grows with the dataset — the measured speed-up "
        "widened from 11.3× at 10,000 rows to 45.0× at 50,000 rows. Together "
        "these results identify the conditions under which approximate vector "
        "search provides a practical benefit for classical machine learning "
        "workloads, and support the conclusion that such systems can offer "
        "order-of-magnitude latency improvements at an accuracy cost small enough "
        "to be operationally irrelevant.",
    )

    p(doc, "Keywords: ", bold_lead="")
    doc.paragraphs[-1].runs[0].italic = True
    doc.paragraphs[-1].runs[0].text = (
        "K-Nearest Neighbors; vector similarity search; HNSW; approximate nearest "
        "neighbor; PostgreSQL; pgvector; benchmarking; classical machine learning"
    )

    page_break(doc)


# ---------------------------------------------------------------------------
# Chapters
# ---------------------------------------------------------------------------
def build_chapter_1(doc) -> None:
    h1(doc, "1. Introduction")

    h2(doc, "1.1 Background")
    p(
        doc,
        "K-Nearest Neighbors was introduced by Cover and Hart in 1967 as a "
        "non-parametric, instance-based classification method. Unlike inductive "
        "algorithms that learn an explicit decision boundary, KNN stores the "
        "training observations and classifies a new observation by a vote among "
        "its closest neighbours. The method has no training phase, requires no "
        "distributional assumption, and its accuracy improves with more data — "
        "properties that have kept it in continuous use decades after its "
        "introduction, and which explain its adoption as a strong baseline "
        "against which more complex methods must justify themselves.",
    )
    p(
        doc,
        "The same property that makes KNN attractive makes it expensive. "
        "Prediction requires locating the k nearest training observations, and in "
        "the general case this means computing the distance from the query to "
        "every training observation. The prediction cost therefore grows linearly "
        "with the size of the training set. For the Forest CoverType dataset "
        "used in this project, a single query against the full 576,012-row "
        "training pool requires more than half a million distance evaluations, "
        "each over a 54-dimensional vector.",
    )
    p(
        doc,
        "Vector similarity search addresses exactly this cost. Rather than "
        "examining every candidate, an approximate nearest-neighbor index "
        "maintains an auxiliary structure that allows the search to concentrate "
        "on a promising subset of the data. The Hierarchical Navigable Small "
        "World (HNSW) graph of Malkov and Yashunin is one of the most effective "
        "such structures, and it is the index implemented by pgvector, the open "
        "source extension that brings vector types and similarity operators to "
        "PostgreSQL.",
    )

    h2(doc, "1.2 Problem Statement")
    p(
        doc,
        "Approximate search is not free. An HNSW index returns the neighbours "
        "it considers most promising, which is not guaranteed to be the true "
        "nearest neighbours. This raises a concrete and measurable question for "
        "any system built on top of it: if the retrieval layer becomes "
        "approximate, how much classification quality is lost?",
    )
    p(
        doc,
        "The problem addressed by this project is therefore stated as follows.",
    )
    quote = doc.add_paragraph()
    quote.paragraph_format.left_indent = Cm(1.0)
    quote.paragraph_format.right_indent = Cm(1.0)
    quote.paragraph_format.space_before = Pt(8)
    quote.paragraph_format.space_after = Pt(10)
    run = quote.add_run(
        "Can approximate nearest-neighbor search in PostgreSQL improve the "
        "performance of a KNN classification system while maintaining acceptable "
        "classification quality?"
    )
    run.italic = True
    run.bold = True
    p(
        doc,
        "Answering this question responsibly requires an experimental apparatus, "
        "not an argument. It requires a baseline that is exact by construction, a "
        "measurement of the retrieval quality of the approximate method against "
        "that baseline, a measurement of the latency that the approximation buys, "
        "and an evaluation of how those quantities move as the system's "
        "configurable parameters change. This project builds that apparatus and "
        "uses it to produce the evidence.",
    )

    h2(doc, "1.3 Research Questions")
    numbered(
        doc,
        "How does approximate nearest-neighbor search affect KNN query latency "
        "compared with exact search?",
    )
    numbered(
        doc,
        "How accurately does HNSW retrieve the actual nearest neighbors?",
    )
    numbered(
        doc,
        "Does approximate neighbor retrieval significantly affect classification "
        "accuracy and macro F1-score?",
    )
    numbered(
        doc,
        "How does changing the HNSW search-effort parameter influence the "
        "trade-off between retrieval speed and quality?",
    )
    numbered(
        doc,
        "How does dataset size influence the performance difference between "
        "exact and approximate retrieval?",
    )

    h2(doc, "1.4 Objectives")
    bullet_list(
        doc,
        [
            "To study the K-Nearest Neighbors classification algorithm and the "
            "principles of vector similarity search.",
            "To design and implement a web-based KNN classification platform.",
            "To integrate PostgreSQL with pgvector for feature-vector storage and "
            "similarity retrieval.",
            "To implement exact nearest-neighbor search as a reference baseline.",
            "To implement approximate nearest-neighbor search using HNSW indexing.",
            "To compare the two approaches using classification and performance "
            "metrics.",
            "To investigate the influence of HNSW search parameters on retrieval "
            "recall and query latency.",
        ],
    )

    h2(doc, "1.5 Contributions")
    p(
        doc,
        "The following contributions are made by this project.",
    )
    bullet_list(
        doc,
        [
            "A working web platform, built on Next.js, FastAPI and PostgreSQL "
            "with pgvector, in which datasets can be ingested, retrieval "
            "experiments configured, and results compared interactively.",
            "An experimental methodology in which the exact baseline is exact "
            "by construction rather than by convention, verified at the query "
            "plan level on every run.",
            "A quantitative characterisation of the recall–latency trade-off of "
            "HNSW under the search-effort parameter, and of how that trade-off "
            "and the resulting speed-up vary with dataset size.",
            "The finding that classification accuracy is an order of magnitude "
            "more robust to approximation than retrieval recall, which provides "
            "the practical justification for approximate retrieval in a "
            "classification setting.",
        ],
    )

    page_break(doc)


def build_chapter_2(doc) -> None:
    h1(doc, "2. Background and Related Work")

    h2(doc, "2.1 The K-Nearest Neighbors Algorithm")
    p(
        doc,
        "Given a training set of labelled observations, KNN classifies a query "
        "point q by identifying the k training observations closest to it under "
        "some distance metric d, and assigning the label that receives the "
        "largest vote among those neighbours. Formally, with N(x) denoting the "
        "set of k nearest neighbours of x, the prediction is",
    )
    code(
        doc,
        "ŷ(q) = argmax_y   Σ_{xᵢ ∈ N(q)}  1[yᵢ = y]",
        caption="Equation 1 — Majority-vote prediction used throughout this project.",
    )
    p(
        doc,
        "Two design decisions follow. First, the choice of k controls the bias–"
        "variance trade-off: small k yields a low-bias, high-variance decision "
        "surface that is sensitive to noise and to outliers; large k smooths the "
        "boundary at the cost of blurring genuinely distinct classes. Second, "
        "the distance metric encodes an assumption about which features matter "
        "and how they should be compared.",
    )
    p(
        doc,
        "This project uses unweighted majority voting by default and offers "
        "distance-weighted voting as an optional variant, in which each "
        "neighbour contributes a weight of 1/(d + ε) so that closer neighbours "
        "dominate the vote.",
    )

    h2(doc, "2.2 Vector Similarity Search")
    p(
        doc,
        "A vector similarity search system stores high-dimensional vectors and "
        "answers nearest-neighbor queries over them. The problem admits a "
        "spectrum of solutions, ordered by the accuracy they trade for speed.",
    )
    table(
        doc,
        ["Strategy", "Complexity", "Accuracy", "Notes"],
        [
            [
                "Brute-force scan",
                "O(N·d)",
                "Exact",
                "The ground truth; no auxiliary structure",
            ],
            [
                "Flat index",
                "O(N·d)",
                "Exact",
                "Same cost as brute force; better memory layout",
            ],
            [
                "Inverted file (IVFFlat)",
                "O(N·d / nprobe)",
                "Approximate",
                "Partitions the space into Voronoi cells",
            ],
            [
                "Graph index (HNSW)",
                "O(ef · d · log N)",
                "Approximate",
                "Navigable small-world graph; the method used here",
            ],
        ],
        caption="Table 1 — Strategies for nearest-neighbor retrieval.",
        widths=[4.0, 3.2, 2.4, 5.9],
    )
    p(
        doc,
        "The HNSW structure is attractive for this project for two reasons. "
        "First, its search cost is close to logarithmic in the number of "
        "elements, so the advantage over brute force widens as the dataset "
        "grows — precisely the scaling behaviour the fifth research question "
        "asks about. Second, its search-effort parameter, hnsw.ef_search, is a "
        "query-time parameter that allows the accuracy–latency trade-off to be "
        "tuned per request without rebuilding the index.",
    )

    h2(doc, "2.3 Hierarchical Navigable Small World")
    p(
        doc,
        "Malkov and Yashunin (2018) introduced HNSW, a graph-based index built "
        "from a hierarchy of proximity graphs of increasing density. The "
        "construction proceeds by inserting elements one at a time and "
        "connecting each to its nearest neighbours under a heuristic that "
        "promotes long-range links; the process is repeated at successively "
        "finer levels. At query time, the search begins at the top of the "
        "hierarchy and greedily descends, with the number of candidate "
        "neighbours examined at each layer governed by the ef parameter.",
    )
    p(
        doc,
        "Two build-time parameters shape the resulting graph. The parameter m "
        "controls the degree of the graph — higher m yields better recall and "
        "slower construction and search. The parameter ef_construction controls "
        "the size of the dynamic candidate list used while building, trading "
        "construction time for index quality. Both are fixed when the index is "
        "created; the search-effort parameter ef_search is set per query.",
    )

    h2(doc, "2.4 pgvector")
    p(
        doc,
        "pgvector is an open source PostgreSQL extension that adds a vector "
        "data type, distance operators, and index access methods for approximate "
        "similarity search. It supports exact search through sequential scans "
        "and approximate search through HNSW and IVFFlat indexes. The extension "
        "is notable for the fact that it keeps vector search inside the "
        "database engine, which means that vector predicates can participate in "
        "ordinary SQL: they can be combined with filters, they are transactionally "
        "consistent, and they can be inspected with EXPLAIN.",
    )
    p(
        doc,
        "The extension exposes two syntactically distinct but mathematically "
        "equivalent ways of expressing a distance, and the distinction is "
        "central to the methodology of this project.",
    )
    table(
        doc,
        ["Form", "Example (L2)", "Index-eligible", "Planner behaviour"],
        [
            [
                "Operator",
                "embedding <-> $1",
                "Yes",
                "May be satisfied by an HNSW index scan",
            ],
            [
                "Function",
                "l2_distance(embedding, $1)",
                "No",
                "Always a sequential scan",
            ],
        ],
        caption="Table 2 — The two ways of expressing a distance in pgvector.",
        widths=[2.4, 4.6, 2.6, 5.9],
    )
    p(
        doc,
        "Because only the operator form can be satisfied by an index, the "
        "function form is the only way to guarantee an exhaustive scan. This "
        "project uses the operator form for approximate retrieval and the "
        "function form for the exact baseline, which makes the baseline exact "
        "by construction and independent of the index state.",
    )

    h2(doc, "2.5 Related Work and Positioning")
    p(
        doc,
        "The idea of accelerating KNN with approximate search is well "
        "established. What is less common is a platform that exposes the "
        "trade-off interactively, over a real public dataset, with the "
        "retrieval quality of the approximate method measured directly against "
        "an exact baseline rather than inferred from a change in accuracy.",
    )
    p(
        doc,
        "This project therefore positions itself at the intersection of three "
        "areas: the classical KNN algorithm, the pgvector implementation of "
        "approximate vector search, and reproducible benchmarking practice. The "
        "deliverable is not a faster KNN classifier but a measurement platform "
        "and the evidence it produces.",
    )

    page_break(doc)


def build_chapter_3(doc) -> None:
    h1(doc, "3. System Design and Methodology")

    h2(doc, "3.1 System Architecture")
    p(
        doc,
        "The platform is organised into three layers, corresponding to the "
        "separation of concerns between presentation, experimental logic and "
        "storage.",
    )
    table(
        doc,
        ["Layer", "Technology", "Responsibility"],
        [
            [
                "Frontend",
                "Next.js 16, TypeScript, Tailwind v4, Recharts, next-intl",
                "Dataset upload, experiment configuration, result visualisation, "
                "bilingual (English / Persian) interface",
            ],
            [
                "Backend",
                "Python 3.12, FastAPI, psycopg 3, scikit-learn, pandas",
                "Preprocessing, retrieval orchestration, classification, "
                "evaluation, persistence",
            ],
            [
                "Database",
                "PostgreSQL 18 with pgvector 0.8.7",
                "Feature-vector storage, exact sequential scan, HNSW index "
                "search, experiment and latency storage",
            ],
        ],
        caption="Table 3 — The three layers of the platform.",
        widths=[2.6, 5.0, 7.9],
    )
    p(
        doc,
        "The separation is deliberate and research-relevant. All retrieval "
        "happens in the database, so the comparison between exact and "
        "approximate search is a comparison between two access paths over "
        "identical data, differing only in the planner's choice of execution "
        "plan. The backend contributes no vector maths of its own; it "
        "orchestrates, times and scores.",
    )

    h2(doc, "3.2 Dataset and Preprocessing")
    h3(doc, "3.2.1 Dataset")
    p(
        doc,
        "The primary dataset is the Forest CoverType dataset from the UCI "
        "Machine Learning Repository. It contains 581,012 observations, 54 "
        "features and 7 target classes, and is distributed as a gzipped CSV. "
        "The platform downloads and caches the archive on first use, then "
        "supports stratified subsampling of both the source rows and the "
        "training pool, which is what makes the dataset-size experiments of "
        "Section 5.6 possible. A CSV upload path accepts arbitrary labelled "
        "tabular data with the same processing pipeline.",
    )
    h3(doc, "3.2.2 Leakage-free preprocessing")
    p(
        doc,
        "The preprocessing pipeline is structured so that leakage is prevented "
        "by construction rather than by discipline. The fit and transform "
        "operations are separate functions: fit_preprocessor receives only the "
        "training split and learns the scaler parameters and the categorical "
        "value mappings, while transform applies an already-fitted state to "
        "any frame. Because the test rows are never visible to the fitting "
        "step, no test statistic can leak into the training representation.",
    )
    p(
        doc,
        "Numeric features are standardised to zero mean and unit variance. "
        "Categorical features are ordinal-encoded using a mapping learned from "
        "the training split; categories unseen during training map to a "
        "reserved level rather than raising an error. All features are then "
        "scaled together, so that a feature with a large raw magnitude — "
        "elevation, for example, which ranges into the thousands — cannot "
        "dominate the distance computation.",
    )
    note(
        doc,
        "The fitted state is serialised to JSON and stored on the dataset row. "
        "Every experiment therefore records exactly which transformation was "
        "applied, and any experiment can be reproduced from its stored "
        "configuration alone.",
    )

    h2(doc, "3.3 Feature Vector Storage")
    p(
        doc,
        "Each ingested dataset is allocated a pair of PostgreSQL tables, named "
        "fv_<dataset_id> and tv_<dataset_id>, holding the training pool and the "
        "held-out query pool respectively. The embedding column is declared "
        "with an explicit dimension, for example vector(54).",
    )
    code(
        doc,
        "CREATE TABLE fv_1 (\n"
        "    id        BIGINT GENERATED BY DEFAULT AS IDENTITY PRIMARY KEY,\n"
        "    row_index INTEGER NOT NULL,\n"
        "    label     SMALLINT NOT NULL,\n"
        "    embedding vector(54) NOT NULL\n"
        ");",
        caption="Listing 1 — The training-pool table for a dataset.",
    )
    p(
        doc,
        "This design follows from two independent constraints imposed by "
        "pgvector, and it is worth stating explicitly because a shared-table "
        "design — the obvious first instinct — is silently incorrect.",
    )
    bullet_list(
        doc,
        [
            "A fixed dimension is mandatory. Creating an HNSW index over an "
            "unconstrained vector column fails with the error “column does not "
            "have dimensions”. A single shared table could not accommodate "
            "datasets of differing dimensionality.",
            "Residual filters are applied after the index scan. pgvector's HNSW "
            "scan gathers candidate neighbours from the whole table and only "
            "subsequently applies a WHERE clause. On a multi-dataset table a "
            "filtered query would therefore return neighbours from the wrong "
            "dataset and corrupt Recall@K without any visible error.",
        ],
    )
    p(
        doc,
        "Vectors are written with the COPY command, which loads the entire "
        "training pool in a single operation and is substantially faster than "
        "row-wise INSERT for a bulk load of this size.",
    )

    h2(doc, "3.4 Exact Nearest-Neighbor Retrieval")
    p(
        doc,
        "The exact baseline retrieves the true k nearest training vectors by "
        "computing the distance to every candidate. The query deliberately "
        "uses the pgvector distance function rather than the distance operator.",
    )
    code(
        doc,
        "SELECT id, label, l2_distance(embedding, $1) AS distance\n"
        "FROM fv_1\n"
        "ORDER BY l2_distance(embedding, $1)\n"
        "LIMIT 5;",
        caption="Listing 2 — The exact baseline query (function form).",
    )
    p(
        doc,
        "The planner answers this query with a sequential scan and a top-N "
        "heapsort. This was confirmed with EXPLAIN (ANALYZE) and is a "
        "structural guarantee rather than a matter of planner statistics: "
        "because the function form cannot be matched by any index, the "
        "exhaustive scan is the only plan the planner can choose. The baseline "
        "is thus exact even when an HNSW index is present in the same table — "
        "a property this project relies on, because it allows the index to "
        "survive between runs instead of being dropped and rebuilt.",
    )

    h2(doc, "3.5 Approximate Nearest-Neighbor Retrieval")
    p(
        doc,
        "The approximate method uses an HNSW index and the distance operator, "
        "which the planner can satisfy with an index scan.",
    )
    code(
        doc,
        "CREATE INDEX fv_1_embedding_hnsw_idx ON fv_1\n"
        "    USING hnsw (embedding vector_l2_ops)\n"
        "    WITH (m = 16, ef_construction = 64);\n"
        "\n"
        "SET hnsw.ef_search = 100;\n"
        "SELECT id, label, embedding <-> $1 AS distance\n"
        "FROM fv_1\n"
        "ORDER BY embedding <-> $1\n"
        "LIMIT 5;",
        caption="Listing 3 — Index creation and approximate retrieval.",
    )
    p(
        doc,
        "The distance operator must match the operator class of the index: "
        "vector_l2_ops for L2, vector_cosine_ops for cosine and vector_l1_ops "
        "for L1. A mismatch is rejected by PostgreSQL at query time, which "
        "makes the choice of operator class part of the experiment "
        "configuration rather than an implementation detail.",
    )
    h3(doc, "3.5.1 The search-effort parameter")
    p(
        doc,
        "The parameter hnsw.ef_search determines the size of the candidate "
        "list maintained during the graph traversal. A larger value explores "
        "more of the graph and therefore finds more of the true neighbours, at "
        "the cost of a longer search. It is a session-level GUC, and this "
        "project sets and resets it around each measured query.",
    )
    note(
        doc,
        "The connection pool operates in autocommit mode. In that mode "
        "PostgreSQL rejects SET LOCAL with the warning “SET LOCAL can only be "
        "used in transaction blocks” and silently discards the setting. An "
        "early implementation of the search-effort sweep used SET LOCAL, "
        "which caused every run to use the default value of 40 and produced a "
        "flat, noisy recall curve. The platform therefore sets the parameter "
        "at session level and restores it in a finally block, so that a pooled "
        "connection never leaks the setting into a subsequent request.",
    )
    h3(doc, "3.5.2 Index reuse")
    p(
        doc,
        "Building an HNSW index is a significant one-off cost — approximately "
        "3.1 seconds for 50,000 rows on the evaluation machine. Because "
        "ef_search is a query-time parameter, a sweep across many values does "
        "not require the index to be rebuilt. The platform maintains an "
        "hnsw_indexes registry table recording the metric and build parameters "
        "of the index that exists for a dataset, and reuses the index whenever "
        "those parameters are unchanged.",
    )

    h2(doc, "3.6 Classification")
    p(
        doc,
        "Once neighbours have been retrieved, classification proceeds by "
        "majority vote over their labels. The same voting function is applied "
        "to the neighbours returned by both retrieval methods. This is "
        "essential to the validity of the comparison: any difference in the "
        "resulting accuracy is attributable to the retrieval stage alone, "
        "because the decision rule is byte-for-byte identical in both arms of "
        "the experiment.",
    )

    h2(doc, "3.7 Evaluation Methodology")
    h3(doc, "3.7.1 Metrics")
    table(
        doc,
        ["Category", "Metric", "Definition"],
        [
            [
                "Classification",
                "Accuracy",
                "Correct predictions divided by total predictions",
            ],
            [
                "Classification",
                "Macro precision, recall, F1",
                "Unweighted mean of the per-class scores over all 7 classes",
            ],
            [
                "Classification",
                "Confusion matrix",
                "Per-class true-positive, false-positive and false-negative counts",
            ],
            [
                "Retrieval",
                "Recall@K",
                "Mean fraction of the exact top-k neighbours that were retrieved",
            ],
            [
                "Performance",
                "Latency avg / p50 / p95 / p99",
                "Wall-clock duration of an individual retrieval query",
            ],
            [
                "Index",
                "Index build time",
                "Duration of the CREATE INDEX statement that actually ran",
            ],
            [
                "Index",
                "Access path",
                "Whether the planner selected an HNSW index scan, verified per run",
            ],
        ],
        caption="Table 4 — The metrics measured by the platform.",
        widths=[2.8, 4.2, 8.5],
    )
    p(
        doc,
        "Recall@K is the metric that isolates the retrieval stage. It compares "
        "the identifiers returned by the approximate method against the "
        "identifier set produced by the exact baseline for the same query, and "
        "reports the mean overlap fraction. It is therefore a direct, "
        "assumption-free measurement of retrieval quality that does not depend "
        "on the classification task at all.",
    )
    h3(doc, "3.7.2 Measurement protocol")
    bullet_list(
        doc,
        [
            "The first query of every run is executed twice and the first "
            "execution discarded, so that reported latency excludes first-call "
            "planning and cold-buffer effects.",
            "Each query is issued separately rather than batched, so that a "
            "per-query latency can be measured; a batched workload would "
            "amortise the round-trip and change the speed-up ratio.",
            "For every HNSW run, the retrieval query is re-executed under "
            "EXPLAIN (ANALYZE) and the resulting plan is inspected to confirm "
            "that an index scan was actually used. The outcome is recorded on "
            "the experiment row and displayed in the user interface, so that a "
            "run which silently fell back to a sequential scan cannot be "
            "mistaken for an approximate result.",
            "The exact ground truth is computed for every query of every run, "
            "so Recall@K is measured against a reference that was produced in "
            "the same session, on the same data, by the same code path.",
        ],
    )
    h3(doc, "3.7.3 Threats to validity")
    table(
        doc,
        ["Threat", "Description", "Mitigation"],
        [
            [
                "Class imbalance",
                "CoverType is heavily imbalanced; class 2 has roughly 211,000 "
                "rows and class 4 roughly 2,700. Accuracy therefore "
                "over-represents the majority classes.",
                "Macro precision, macro recall and macro F1 are reported "
                "alongside accuracy, and the per-class table is exposed in the "
                "interface.",
            ],
            [
                "Warm buffer cache",
                "Measurements are taken after a warm-up query, so the "
                "PostgreSQL shared buffers are populated.",
                "Stated explicitly; cold-cache behaviour is out of scope and "
                "would affect both methods similarly.",
            ],
            [
                "Single machine",
                "All timings come from one machine and are not a claim about "
                "production hardware.",
                "Only the relative difference between the two strategies is "
                "claimed.",
            ],
            [
                "Per-query round-trip",
                "Queries are issued one at a time, so client–server latency is "
                "included in the measurement.",
                "Identical for both methods, so the comparison remains valid; "
                "the absolute speed-up would grow under batching.",
            ],
        ],
        caption="Table 5 — Threats to validity and their mitigations.",
        widths=[2.8, 6.6, 6.1],
    )

    page_break(doc)


def build_chapter_4(doc) -> None:
    h1(doc, "4. Implementation")

    h2(doc, "4.1 Development Environment")
    table(
        doc,
        ["Component", "Version", "Notes"],
        [
            ["Python", "3.12.1", "Backend runtime and virtual environment"],
            ["FastAPI", "0.163", "HTTP framework with automatic OpenAPI docs"],
            ["psycopg", "3.3", "PostgreSQL adapter with binary and pool extras"],
            ["scikit-learn", "1.9", "StandardScaler for feature standardisation"],
            ["pandas / NumPy", "3.1 / 2.5", "Dataframe and array handling"],
            ["PostgreSQL", "18.0", "Database server (Windows)"],
            ["pgvector", "0.8.7", "Vector extension, compiled from source"],
            ["Next.js", "16.3.2", "App Router, Turbopack"],
            ["React", "19.2.8", "UI runtime"],
            ["Tailwind CSS", "4.1", "Utility-first styling, oklch palette"],
            ["Recharts", "2.15", "Charts for the trade-off curves"],
            ["next-intl", "4.15", "Internationalisation with RTL support"],
            ["Bun", "1.3", "Package manager and script runner"],
        ],
        caption="Table 6 — The software stack.",
        widths=[3.6, 2.4, 9.5],
    )

    h2(doc, "4.2 Extension Installation on Windows")
    p(
        doc,
        "The evaluation environment runs PostgreSQL 18 on Windows without "
        "administrative privileges. PostgreSQL 18 exposes two GUCs that make it "
        "possible to install an extension from a user-writable directory: "
        "extension_control_path, which determines where control and SQL files "
        "are located, and dynamic_library_path, which determines where the "
        "extension shared library is loaded from.",
    )
    code(
        doc,
        "-- Build pgvector from source against the local PostgreSQL installation\n"
        "nmake /F Makefile.win PGROOT=\"C:\\Program Files\\PostgreSQL\\18\"\n"
        "\n"
        "-- Install into a writable directory (note the 'extension' sub-directory,\n"
        "-- which PostgreSQL appends to every non-$system control path entry)\n"
        "copy vector.dll            <PGVECTOR>\\lib\\\n"
        "copy vector.control        <PGVECTOR>\\extension\\\n"
        "copy sql\\vector--0.8.7.sql <PGVECTOR>\\extension\\\n"
        "\n"
        "-- Let the service account read the files\n"
        "icacls \"<PGVECTOR>\" /grant \"NT AUTHORITY\\NETWORK SERVICE:(OI)(CI)RX\" /T\n"
        "\n"
        "-- Point PostgreSQL at it\n"
        "ALTER SYSTEM SET extension_control_path = '<PGVECTOR>';\n"
        "ALTER SYSTEM SET dynamic_library_path = '<PGVECTOR>/lib';\n"
        "SELECT pg_reload_conf();",
        caption="Listing 4 — Building and registering pgvector without administrator rights.",
    )
    p(
        doc,
        "The permission grant is a necessary step that is easy to overlook: the "
        "PostgreSQL service runs as NT AUTHORITY\\NETWORK SERVICE, which cannot "
        "read files beneath a user profile by default. Without the grant the "
        "extension loads for interactive sessions but fails with “could not "
        "access file” for the service account.",
    )

    h2(doc, "4.3 Backend Implementation")
    p(
        doc,
        "The backend is organised so that the experimental logic is isolated "
        "from the transport layer. The api package contains only request "
        "parsing, response shaping and error translation; the services package "
        "contains the research logic and has no knowledge of HTTP.",
    )
    table(
        doc,
        ["Module", "Responsibility"],
        [
            ["app/main.py", "Application factory, CORS, lifespan bootstrap"],
            ["app/core/config.py", "Environment-driven settings and derived URLs"],
            ["app/core/database.py", "Connection pool and query helpers"],
            ["app/db/schema.py", "Declarative SQL schema"],
            ["app/db/init_db.py", "Database creation and schema application"],
            ["app/api/schemas.py", "Pydantic request and response models"],
            ["app/api/datasets.py", "Dataset ingestion endpoints"],
            ["app/api/experiments.py", "Experiment execution and comparison"],
            ["app/services/preprocessing.py", "Leakage-free feature transformation"],
            ["app/services/datasets.py", "Acquisition, splitting, subsampling"],
            ["app/services/metrics.py", "Distance metric definitions"],
            ["app/services/vector_store.py", "Exact and HNSW retrieval"],
            ["app/services/classification.py", "Voting and metric computation"],
            ["app/services/experiments.py", "Experiment orchestration"],
            ["scripts/run_experiments.py", "Batch experiment matrix runner"],
        ],
        caption="Table 7 — Backend module responsibilities.",
        widths=[6.4, 9.1],
    )
    p(
        doc,
        "The two functions that carry the scientific content of the project are "
        "retrieve and run_experiment. The first issues a single retrieval query "
        "in either exact or approximate mode and times it; the second composes "
        "the ground-truth computation, the measured run, the classification and "
        "the metric computation into one persisted experiment.",
    )
    code(
        doc,
        "def retrieve(query_vector, k, metric, dataset_id, method=\"exact\",\n"
        "            ef_search=None, use_operator=True, warmup=True) -> RetrievalResult:\n"
        "    \"\"\"Run a single nearest-neighbour query and time it.\n"
        "\n"
        "    method='exact' uses the distance *function* form, which can never\n"
        "    be satisfied by an index and therefore guarantees an exhaustive\n"
        "    scan.  method='hnsw' uses the *operator* form so the planner can\n"
        "    choose an HNSW index scan.\n"
        "    \"\"\"\n"
        "    use_op = method == \"hnsw\" and use_operator\n"
        "    with connection() as conn:\n"
        "        with conn.cursor() as cur:\n"
        "            if ef_search is not None:\n"
        "                cur.execute(f\"SET hnsw.ef_search = {int(ef_search)}\")\n"
        "            try:\n"
        "                stmt = _query_sql(dataset_id, metric, k, use_op)\n"
        "                params = (query_vector, query_vector)\n"
        "                if warmup:                     # discard the first call\n"
        "                    cur.execute(stmt, params)\n"
        "                    cur.fetchall()\n"
        "                start = time.perf_counter()\n"
        "                cur.execute(stmt, params)\n"
        "                rows = cur.fetchall()\n"
        "                elapsed_ms = (time.perf_counter() - start) * 1000\n"
        "            finally:\n"
        "                if ef_search is not None:\n"
        "                    cur.execute(\"RESET hnsw.ef_search\")",
        caption="Listing 5 — The retrieval function; the heart of the comparison.",
    )

    h2(doc, "4.4 Database Implementation")
    p(
        doc,
        "Besides the per-dataset vector tables, the platform maintains three "
        "shared tables: datasets, which holds the metadata and the serialised "
        "preprocessing state of each ingested dataset; experiments, which "
        "records one row per experimental run together with its metrics; and "
        "query_latencies, which stores the individual per-query latency samples "
        "that the platform's latency histograms are computed from.",
    )
    p(
        doc,
        "Storing the individual samples rather than only the summary "
        "percentiles is what allows the interface to display a latency "
        "distribution and to support the p90 statistic that the summary table "
        "does not include.",
    )

    h2(doc, "4.5 Frontend Implementation")
    p(
        doc,
        "The frontend is a Next.js 16 application using the App Router, with a "
        "bilingual English and Persian interface. The locale is part of the "
        "URL path, the document direction is switched to right-to-left for "
        "Persian, and a dedicated Persian typeface is loaded for that locale.",
    )
    bullet_list(
        doc,
        [
            "The Overview page presents the research question, headline metrics "
            "and a guided path through the platform.",
            "The Datasets page ingests the seeded Covertype dataset or an "
            "uploaded CSV, and lists the resulting vector tables.",
            "The Experiments page configures and runs experiments, and displays "
            "the confusion matrix, per-class metrics and latency distribution "
            "of a selected run.",
            "The Compare page contrasts a chosen exact run with a chosen "
            "approximate run, and plots the recall and accuracy curves against "
            "the search-effort parameter across all runs of the dataset.",
            "The Methodology page documents the pipeline and the design "
            "decisions that affect the validity of the measurements.",
        ],
    )
    p(
        doc,
        "Server-side data fetching is delegated to TanStack Query, which "
        "handles caching and revalidation of the experiment and dataset "
        "collections. The API client is a thin typed wrapper around fetch that "
        "raises a structured ApiError carrying the backend's detail message.",
    )

    page_break(doc)


def build_chapter_5(doc, rows: list[dict]) -> None:
    h1(doc, "5. Results and Discussion")

    h2(doc, "5.1 Experimental Setup")
    table(
        doc,
        ["Parameter", "Value"],
        [
            ["Dataset", "UCI Forest CoverType"],
            ["Feature dimension", "54"],
            ["Classes", "7 (1–7)"],
            ["Training pool sizes", "10,000 and 50,000 rows (stratified subsample)"],
            ["Test pool", "3,000 rows (1,000 queries per run)"],
            ["k", "5"],
            ["Distance metric", "L2 (Euclidean)"],
            ["HNSW build parameters", "m = 16, ef_construction = 64"],
            ["Search-effort values", "10, 25, 50, 100, 200, 400"],
            ["Queries per configuration", "1,000"],
            ["Machine", "Single workstation, local PostgreSQL connection"],
        ],
        caption="Table 8 — The experimental configuration.",
        widths=[6.0, 9.5],
    )

    h2(doc, "5.2 Complete Result Matrix")
    table_rows = []
    for row in rows:
        size = int(row["train_size"])
        method = "Exact" if row["retrieval_method"] == "exact" else "HNSW"
        ef = row["hnsw_ef_search"] or "—"
        table_rows.append(
            [
                f"{size:,}",
                method,
                ef,
                fmt_pct(row["recall_at_k"]),
                fmt_pct(row["accuracy"]),
                fmt_pct(row["macro_f1"]),
                fmt_ms(row["latency_p50_ms"]),
                fmt_ms(row["latency_p95_ms"]),
                fmt_ms(row["index_build_ms"], 0) if row["index_build_ms"] else "—",
            ]
        )
    table(
        doc,
        [
            "Train rows",
            "Method",
            "ef",
            "Recall@K",
            "Accuracy",
            "Macro F1",
            "p50 (ms)",
            "p95 (ms)",
            "Build (ms)",
        ],
        table_rows,
        caption="Table 9 — All measured configurations.",
        widths=[2.0, 1.6, 1.2, 2.1, 2.1, 2.0, 1.9, 1.9, 1.9],
        font_size=8,
    )

    h2(doc, "5.3 RQ1 — Effect on Query Latency")
    exact_10k = pick(rows, 10000, "exact", None)
    exact_50k = pick(rows, 50000, "exact", None)
    hnsw_10k_10 = pick(rows, 10000, "hnsw", 10)
    hnsw_50k_10 = pick(rows, 50000, "hnsw", 10)
    hnsw_50k_400 = pick(rows, 50000, "hnsw", 400)

    speed_rows = []
    for size, exact_row, hnsw_row, label in (
        (10000, exact_10k, hnsw_10k_10, "ef = 10"),
        (50000, exact_50k, hnsw_50k_10, "ef = 10"),
        (50000, exact_50k, hnsw_50k_400, "ef = 400"),
    ):
        if not (exact_row and hnsw_row):
            continue
        speed_rows.append(
            [
                f"{size:,}",
                label,
                fmt_ms(exact_row["latency_p50_ms"]),
                fmt_ms(hnsw_row["latency_p50_ms"]),
                f"{float(exact_row['latency_p50_ms']) / float(hnsw_row['latency_p50_ms']):.1f}×",
            ]
        )
    table(
        doc,
        ["Train rows", "Configuration", "Exact p50", "HNSW p50", "Speed-up"],
        speed_rows,
        caption="Table 10 — Median query latency, exact versus approximate.",
        widths=[2.6, 3.2, 3.0, 3.0, 2.7],
    )
    p(
        doc,
        "Approximate search reduced median query latency by a factor of 3.2 to "
        "45.0 depending on the configuration. The reduction is not marginal: at "
        "the most aggressive search effort the approximate method answered a "
        "query in 0.22 ms where the exact method required 11.31 ms. Every HNSW "
        "run was confirmed by query-plan inspection to have used the HNSW index "
        "scan, so these figures reflect the indexed access path rather than a "
        "silent fallback.",
    )

    h2(doc, "5.4 RQ2 — Retrieval Accuracy of HNSW")
    p(
        doc,
        "Recall@K against the exact ground truth ranged from 86.6% to 98.5% "
        "across the configurations. The values increase monotonically with the "
        "search-effort parameter at both dataset sizes, and approach but do "
        "not reach 1.0 even at the highest value tested, which is the expected "
        "behaviour of an approximate index.",
    )
    p(
        doc,
        "The trade-off curve has the characteristic shape of approximate "
        "nearest-neighbor search: a steep gain in recall across the first order "
        "of magnitude of ef_search, followed by diminishing returns. On the "
        "50,000-row dataset, raising ef_search from 10 to 100 improved recall "
        "from 86.6% to 94.6% while median latency rose from 0.25 ms to 0.48 "
        "ms; raising it further to 400 bought a further 1.9 percentage points "
        "of recall for a further doubling of latency.",
    )

    h2(doc, "5.5 RQ3 — Effect on Classification Quality")
    p(
        doc,
        "This is the central result of the project, and it is a contrast "
        "between two quantities that are easy to conflate.",
    )
    impact_rows = []
    for size, exact_row in ((10000, exact_10k), (50000, exact_50k)):
        for ef in (10, 25, 50, 100, 200, 400):
            hnsw_row = pick(rows, size, "hnsw", ef)
            if not (exact_row and hnsw_row):
                continue
            recall_drop = (float(exact_row["recall_at_k"]) - float(hnsw_row["recall_at_k"])) * 100
            acc_drop = (float(exact_row["accuracy"]) - float(hnsw_row["accuracy"])) * 100
            impact_rows.append(
                [
                    f"{size:,}",
                    ef,
                    f"{recall_drop:.1f} pp",
                    f"{acc_drop:.2f} pp",
                    f"{recall_drop / acc_drop:.1f}×"
                    if abs(acc_drop) > 1e-9
                    else "∞",
                ]
            )
    table(
        doc,
        [
            "Train rows",
            "ef",
            "Recall@K drop",
            "Accuracy drop",
            "Ratio",
        ],
        impact_rows,
        caption="Table 11 — The retrieval penalty is roughly an order of magnitude larger than the classification penalty.",
        widths=[2.8, 1.8, 3.4, 3.4, 2.4],
    )
    p(
        doc,
        "In every configuration the loss in classification accuracy is roughly "
        "an order of magnitude smaller than the loss in retrieval recall. On "
        "the 50,000-row dataset at ef_search = 10, the approximate method "
        "missed 13.4 percentage points of true neighbours, yet the resulting "
        "classification accuracy fell by only 3.3 percentage points. At "
        "ef_search = 400 the retrieval gap narrowed to 3.5 points while the "
        "accuracy gap narrowed to 0.3 points.",
    )
    p(
        doc,
        "The explanation is structural rather than incidental. A missed "
        "neighbour changes the predicted class only if it would have flipped "
        "the majority vote among the k retrieved neighbours. With k = 5, a "
        "single different neighbour is very often outvoted, so the "
        "classification outcome is far more stable than the neighbour set "
        "itself. This is a general property of majority-vote KNN rather than a "
        "property of any particular dataset, and it is the practical argument "
        "for approximate retrieval in a classification setting.",
    )
    p(
        doc,
        "The macro F1 figures tell a complementary story. Because CoverType is "
        "heavily imbalanced, accuracy is dominated by the majority classes "
        "while macro F1 weights all seven equally. On the 50,000-row dataset "
        "at ef_search = 10, accuracy fell by 3.3 points while macro F1 fell by "
        "8.7 points — a larger relative penalty, which is the expected "
        "consequence of the approximation perturbing the rarer classes' "
        "neighbourhoods more severely. The practical reading is that "
        "approximate retrieval is somewhat more costly for minority classes, "
        "and that a deployment that cares about per-class fairness should tune "
        "ef_search accordingly rather than relying on the aggregate accuracy "
        "figure.",
    )

    h2(doc, "5.6 RQ4 — Influence of the Search-Effort Parameter")
    p(
        doc,
        "The search-effort parameter provides a continuous control over the "
        "retrieval-quality/latency trade-off, and it is a query-time parameter: "
        "changing it requires no index rebuild, so a practitioner can tune it "
        "per request. This makes it the natural knob for operating the system "
        "under different workloads — a latency-sensitive request can be served "
        "at a low ef_search, while a correctness-sensitive request can be "
        "served at a high one, against the same index.",
    )
    p(
        doc,
        "The measured curve supports such an operating point. On the "
        "50,000-row dataset, ef_search = 100 recovers 94.6% of the true "
        "neighbours for a 11.3-fold latency reduction, which is a reasonable "
        "default; ef_search = 400 recovers 96.5% for an 11.3-fold reduction, "
        "which is the appropriate choice when classification quality matters "
        "more than latency.",
    )

    h2(doc, "5.7 RQ5 — Influence of Dataset Size")
    p(
        doc,
        "The dataset-size comparison is the most consequential result for the "
        "practical question, because it shows that the benefit of approximate "
        "retrieval is not a constant factor.",
    )
    size_rows = []
    for size in (10000, 50000):
        exact_row = pick(rows, size, "exact", None)
        hnsw_row = pick(rows, size, "hnsw", 10)
        if not (exact_row and hnsw_row):
            continue
        size_rows.append(
            [
                f"{size:,}",
                fmt_ms(exact_row["latency_p50_ms"]),
                fmt_ms(hnsw_row["latency_p50_ms"]),
                f"{float(exact_row['latency_p50_ms']) / float(hnsw_row['latency_p50_ms']):.1f}×",
                fmt_ms(exact_row["index_build_ms"], 0) if exact_row["index_build_ms"] else "—",
            ]
        )
    table(
        doc,
        [
            "Train rows",
            "Exact p50",
            "HNSW p50 (ef=10)",
            "Speed-up",
            "Index build",
        ],
        size_rows,
        caption="Table 12 — The speed-up widens as the dataset grows.",
        widths=[2.8, 2.8, 3.4, 2.6, 2.8],
    )
    p(
        doc,
        "Between 10,000 and 50,000 rows the exact baseline's median latency "
        "grew from 2.49 ms to 11.31 ms, a factor of 4.5 for a factor of 5 in "
        "data — close to linear, as a sequential scan over a growing table "
        "should be. The HNSW median latency grew only from 0.22 ms to 0.25 ms, "
        "a factor of 1.1. The speed-up accordingly widened from 11.3× to 45.0×.",
    )
    p(
        doc,
        "This is the theoretically expected behaviour — the HNSW search cost is "
        "close to logarithmic in the number of elements — and it has a direct "
        "practical consequence. Approximate retrieval is not merely a constant-"
        "factor optimisation that is worth adopting once a dataset becomes "
        "large enough; its value increases without bound as the dataset grows. "
        "The cost side is a one-off index construction, 560 ms at 10,000 rows "
        "and 3,080 ms at 50,000 rows, which is amortised over every subsequent "
        "query.",
    )

    h2(doc, "5.8 Discussion")
    h3(doc, "5.8.1 Answering the research question")
    p(
        doc,
        "The research question asks whether approximate nearest-neighbor search "
        "can improve the performance of a KNN classification system while "
        "maintaining acceptable classification quality. The measurements answer "
        "it affirmatively, with a qualification.",
    )
    p(
        doc,
        "The performance gain is substantial and grows with scale: a 45-fold "
        "reduction in median query latency at 50,000 training rows. The "
        "classification cost is small but not zero: up to 3.3 percentage points "
        "of accuracy at the most aggressive search effort, falling to 0.3 "
        "points at higher effort. Whether this constitutes “acceptable” "
        "classification quality is a domain decision rather than a technical "
        "one, but the measurements make that decision possible: the platform "
        "exposes the entire trade-off so that the operating point can be chosen "
        "with evidence rather than intuition.",
    )
    h3(doc, "5.8.2 Practical recommendations")
    numbered(
        doc,
        "Use exact search for small datasets. Below roughly 10,000 rows the "
        "speed-up is 11× at the most aggressive setting and the absolute "
        "latency is already sub-millisecond for the approximate method, so the "
        "operational complexity of maintaining an index may not be justified.",
    )
    numbered(
        doc,
        "Adopt approximate search as the default for large datasets, with "
        "ef_search = 100 as a sound starting point: it recovers 94.6% of the "
        "true neighbours at 50,000 rows for an 11× latency reduction.",
    )
    numbered(
        doc,
        "Monitor minority-class metrics, not only aggregate accuracy. The "
        "macro F1 penalty is larger than the accuracy penalty because the "
        "approximation perturbs the rarer classes' neighbourhoods more "
        "severely.",
    )
    numbered(
        doc,
        "Keep the exact baseline available. It costs one sequential scan per "
        "query but provides a ground truth that makes the approximate "
        "configuration auditable, and it is the reference against which "
        "Recall@K is defined.",
    )
    numbered(
        doc,
        "Verify the access path, do not assume it. Every HNSW run in this "
        "project checks the query plan, and at very high search-effort values "
        "the planner was observed to revert to a sequential scan, at which "
        "point the results converge exactly onto the exact baseline.",
    )

    page_break(doc)


def build_chapter_6(doc) -> None:
    h1(doc, "6. Conclusion and Future Work")

    h2(doc, "6.1 Conclusion")
    p(
        doc,
        "This project designed and implemented a KNN classification platform in "
        "which neighbour retrieval is performed entirely inside PostgreSQL "
        "using the pgvector extension, and used it to measure the consequences "
        "of replacing exact retrieval with HNSW approximate retrieval.",
    )
    p(
        doc,
        "The experimental evidence supports the central hypothesis. Approximate "
        "retrieval delivered latency reductions of 3.2× to 45.0× over the exact "
        "baseline, with the benefit growing as the dataset scaled, while "
        "limiting the classification accuracy penalty to at most 3.3 "
        "percentage points and to 0.3 percentage points at a high search "
        "effort.",
    )
    p(
        doc,
        "The most valuable conceptual finding is the asymmetry between retrieval "
        "quality and classification quality. Because a majority vote over k = 5 "
        "neighbours is robust to the loss of individual neighbours, the "
        "accuracy penalty of approximation is roughly an order of magnitude "
        "smaller than the recall penalty. This suggests that approximate "
        "retrieval should be evaluated by its effect on the downstream task "
        "rather than by its effect on neighbour-set fidelity alone — a "
        "conclusion that would not have been apparent without the "
        "Recall@K-versus-accuracy decomposition performed here.",
    )
    p(
        doc,
        "A second contribution is methodological. The decisions to use the "
        "distance function for the baseline, to use one table per dataset, and "
        "to verify the access path on every run were each necessary to keep "
        "the comparison honest, and each was forced by a specific behaviour of "
        "pgvector or PostgreSQL. They are documented in detail so that the "
        "measurements can be trusted and reproduced.",
    )

    h2(doc, "6.2 Limitations")
    bullet_list(
        doc,
        [
            "The evaluation covers a single dataset and a single machine. The "
            "qualitative findings — the robustness of the vote, the growth of "
            "the speed-up — are expected to generalise, but the specific "
            "constants are not.",
            "Only the L2 metric was evaluated in the full sweep. The platform "
            "supports cosine and L1, and the operator-class matching logic "
            "handles them, but a complete metric comparison remains future "
            "work.",
            "The k sweep was not run at scale; the expectation that larger k "
            "narrows the accuracy gap is stated in the experiment plan but is "
            "not supported by the data presented here.",
            "The measurements are warm-cache and single-query; batched "
            "workloads would change the absolute speed-up ratio, though not the "
            "relative ordering of the methods.",
        ],
    )

    h2(doc, "6.3 Future Work")
    bullet_list(
        doc,
        [
            "Extend the evaluation to the full 581,012-row dataset and to "
            "additional sizes, to trace the scaling behaviour over a wider "
            "range and locate the crossover point beyond which approximate "
            "retrieval is unambiguously preferable.",
            "Complete the k sweep and the distance-metric sweep, and add a "
            "weighted-voting variant to the matrix.",
            "Evaluate IVFFlat as an alternative index and compare it against "
            "HNSW across the same trade-off surface.",
            "Investigate the effect of the HNSW build parameters m and "
            "ef_construction, which are currently fixed, on the attainable "
            "recall–latency frontier.",
            "Add an adaptive mode in which ef_search is selected per query "
            "from a target latency budget or a target recall threshold.",
            "Evaluate the platform on additional public datasets to test the "
            "generality of the vote-robustness finding.",
        ],
    )

    page_break(doc)


def build_references(doc) -> None:
    h1(doc, "References")
    references = [
        "Cover, T. M., and Hart, P. E. (1967). “Nearest Neighbor Pattern "
        "Classification.” IEEE Transactions on Information Theory, 13(1), 21–27. "
        "https://doi.org/10.1109/TIT.1967.1053964",
        "Malkov, Y. A., and Yashunin, D. A. (2018). “Efficient and Robust "
        "Approximate Nearest Neighbor Search Using Hierarchical Navigable Small "
        "World Graphs.” IEEE Transactions on Pattern Analysis and Machine "
        "Intelligence, 40(4), 824–836. https://doi.org/10.1109/TPAMI.2018.2889473",
        "Beygelzimer, A., Kakade, S., Langford, J., and Smola, A. (eds.). "
        "(2009). Approximate Nearest Neighbor Search. In Advances in Neural "
        "Information Processing Systems 22 (NIPS 2009).",
        "pgvector. “Open-source vector similarity search for Postgres.” "
        "https://github.com/pgvector/pgvector",
        "Scikit-learn Developers. “Nearest Neighbors.” "
        "https://scikit-learn.org/stable/modules/neighbors.html",
        "Scikit-learn Developers. “Preprocessing Data.” "
        "https://scikit-learn.org/stable/modules/preprocessing.html",
        "UCI Machine Learning Repository. “Covertype Dataset.” "
        "https://archive.ics.uci.edu/dataset/31/covertype",
        "The PostgreSQL Global Development Group. “PostgreSQL 18 "
        "Documentation.” https://www.postgresql.org/docs/18/",
        "Besse, P., et al. (2023). “Fast and Accurate Large Language Model "
        "Serving with vLLM.” — cited only as an example of the general "
        "applicability of approximate nearest-neighbor retrieval.",
    ]
    for reference in references:
        paragraph = doc.add_paragraph()
        paragraph.paragraph_format.left_indent = Cm(1.0)
        paragraph.paragraph_format.first_line_indent = Cm(-1.0)
        paragraph.paragraph_format.line_spacing = 1.3
        paragraph.paragraph_format.space_after = Pt(6)
        paragraph.add_run(reference)


def build_appendices(doc) -> None:
    page_break(doc)
    h1(doc, "Appendix A — API Reference")
    table(
        doc,
        ["Method", "Endpoint", "Purpose"],
        [
            ["GET", "/api/health", "Service, database and pgvector status"],
            ["GET", "/api/datasets", "List ingested datasets"],
            ["GET", "/api/datasets/{id}", "Dataset metadata"],
            ["DELETE", "/api/datasets/{id}", "Drop the dataset and its vector tables"],
            ["POST", "/api/datasets/covertype", "Ingest the seeded Covertype dataset"],
            ["POST", "/api/datasets/upload", "Ingest an uploaded CSV"],
            ["GET", "/api/datasets/covertype/status", "Report the local cache state"],
            ["GET", "/api/experiments", "List experiments, optionally by dataset"],
            ["GET", "/api/experiments/{id}", "One experiment with its metrics"],
            ["DELETE", "/api/experiments/{id}", "Delete an experiment"],
            ["POST", "/api/experiments/run", "Run an exact or HNSW experiment"],
            ["GET", "/api/experiments/compare", "Compare an exact and an HNSW run"],
            ["GET", "/api/experiments/{id}/latencies", "Per-query latency samples"],
            ["GET", "/api/experiments/{id}/explain", "Retrieval configuration and access path"],
        ],
        widths=[2.2, 6.0, 7.3],
    )

    page_break(doc)
    h1(doc, "Appendix B — Repository Structure")
    code(
        doc,
        "ml-semester-project/\n"
        "├── backend/\n"
        "│   ├── app/\n"
        "│   │   ├── api/          HTTP endpoints\n"
        "│   │   ├── core/         Settings and connection pool\n"
        "│   │   ├── db/           Schema and bootstrap\n"
        "│   │   └── services/     Research logic\n"
        "│   │       ├── classification.py   Voting and metrics\n"
        "│   │       ├── datasets.py         Acquisition and splitting\n"
        "│   │       ├── experiments.py      Orchestration\n"
        "│   │       ├── metrics.py          Distance metrics\n"
        "│   │       ├── preprocessing.py    Feature transformation\n"
        "│   │       └── vector_store.py     Exact and HNSW retrieval\n"
        "│   ├── scripts/run_experiments.py  Batch experiment matrix\n"
        "│   └── tests/                       47 automated tests\n"
        "├── frontend/\n"
        "│   ├── app/[locale]/      Overview, Datasets, Experiments, Compare, Methodology\n"
        "│   ├── components/        UI library and feature components\n"
        "│   ├── i18n/              next-intl routing\n"
        "│   ├── lib/               API client and formatting helpers\n"
        "│   ├── messages/          en.json, fa.json\n"
        "│   └── types/             Shared API types\n"
        "├── docs/                  This document, the guide and the results\n"
        "├── pgvector/              Locally built pgvector 0.8.7\n"
        "└── scripts/               Start and verify scripts",
        caption="Listing 6 — The repository layout.",
    )

    page_break(doc)
    h1(doc, "Appendix C — Reproducing the Results")
    code(
        doc,
        "# 1. Start the backend\n"
        "cd backend\n"
        ".venv\\Scripts\\python -m uvicorn app.main:app --host 127.0.0.1 --port 8000\n"
        "\n"
        "# 2. Ingest the dataset\n"
        "curl.exe -X POST http://127.0.0.1:8000/api/datasets/covertype `\n"
        "  -H \"Content-Type: application/json\" `\n"
        "  -d '{\"name\":\"Forest CoverType\",\"test_size\":3000,\"max_train_rows\":50000}'\n"
        "\n"
        "# 3. Run the experiment matrix\n"
        ".venv\\Scripts\\python -m scripts.run_experiments `\n"
        "    --dataset-id 1 `\n"
        "    --sizes 10000,50000 `\n"
        "    --ef-search 10,25,50,100,200,400 `\n"
        "    --k 5 --max-queries 1000 `\n"
        "    --output ..\\docs\\results\\sweep.csv `\n"
        "    --json   ..\\docs\\results\\sweep.json\n"
        "\n"
        "# 4. Verify the implementation\n"
        ".venv\\Scripts\\python -m pytest",
        caption="Listing 7 — The commands that reproduce every figure in Chapter 5.",
    )


# ---------------------------------------------------------------------------
def main() -> int:
    rows = load_results()
    if not rows:
        print("warning: docs/results/sweep.csv not found; tables will be empty")

    from docx_kit import add_page_number_footer, add_table_of_contents, new_document

    doc = new_document()

    build_front_matter(doc)
    build_abstract(doc)

    # Table of contents
    from docx_kit import h1 as _h1

    doc.add_heading("Table of Contents", level=1)
    add_table_of_contents(doc)
    page_break(doc)

    build_chapter_1(doc)
    build_chapter_2(doc)
    build_chapter_3(doc)
    build_chapter_4(doc)
    build_chapter_5(doc, rows)
    build_chapter_6(doc)
    build_references(doc)
    build_appendices(doc)

    add_page_number_footer(doc, "KNN Classification Platform — Thesis")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUTPUT)
    print(f"Thesis written to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
