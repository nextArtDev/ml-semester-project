# Complete Project Guide — Key Concepts and Code Walkthrough

> Written in plain English so it renders cleanly. Keep this as your study reference.

---

## Part 1: What Is This Project About?

### The research question

**Can approximate nearest-neighbor search in PostgreSQL speed up KNN
classification without sacrificing accuracy?**

You have a dataset where each row is a feature vector (a list of numbers).
You want to classify a new, unseen vector by looking at its k closest
neighbors in the training set and taking a majority vote on their labels.
That is KNN (k-nearest neighbors).

The problem: finding the "k closest neighbors" requires computing the
distance from the query vector to *every* vector in the training set, then
sorting them. If you have 50,000 training vectors, that is 50,000 distance
computations per query. It works, but it is slow.

There exists a shortcut: instead of scanning every row, you can build a
**graph index** (called HNSW) that lets you jump around the vectors in a
smart way and find the approximate nearest neighbors much faster — but it
might miss a few.

This project asks: **is the approximation good enough that classification
accuracy stays the same, while the speed goes up dramatically?**

### Why this fits your course constraints

- **No neural networks.** The only "learning" is: store the training
  vectors, then at query time find the nearest ones and vote. There is no
  training phase, no weights, no backpropagation, no gradient descent.
- **KNN is a classic ML algorithm.** It is covered in every ML textbook
  (e.g., "Introduction to Statistical Learning", Chapter 2).
- **Vector databases are the infrastructure that makes KNN practical at
  scale.** Without an efficient retrieval mechanism, KNN is too slow for
  real-world use. The connection between the two is genuine and
  well-motivated.

---

## Part 2: Key Concepts (Glossary)

### KNN (k-Nearest Neighbors)

A **lazy, non-parametric** classification algorithm. "Lazy" means it does
not build a model during training — it just stores the data. "Non-parametric"
means it makes no assumptions about the data distribution. To classify a
new point, it finds the k closest training points and takes a majority vote.

In this project: k=5, distance = L2 (Euclidean), voting = simple majority
(not distance-weighted by default).

### Feature vector

A row of numeric features represented as a point in space. For the
Covertype dataset, each row has 54 features (10 continuous + 44 binary), so
each vector lives in a 54-dimensional space.

### Vector database

A database optimized for storing and querying vectors. Traditional
databases index on exact matches (e.g., "find user with id=42"). Vector
databases index on *similarity* (e.g., "find the 5 vectors closest to this
one"). In this project, PostgreSQL with the pgvector extension acts as the
vector database.

### pgvector

A PostgreSQL extension that adds a `vector` column type and distance
operators. It lets you store vectors directly in database tables and run
similarity queries in SQL.

Key things pgvector provides:
- `vector(dim)` column type (e.g., `vector(54)` for 54-dimensional vectors)
- Distance functions: `l2_distance(a, b)`, `cosine_distance(a, b)`, `l1_distance(a, b)`
- Distance operators: `<->` (L2), `<=>` (cosine), `<+>` (L1)
- Index types: HNSW (used here) and IVFFlat (not used here)

### HNSW (Hierarchical Navigable Small World)

A **graph-based** index for approximate nearest-neighbor search. Instead of
comparing the query to every vector, it navigates a multi-layer graph to
quickly converge on the neighborhood of the query.

Key parameters:
- **m** (default 16): number of bidirectional connections per node. Higher
  = more accurate but more memory and slower build.
- **ef_construction** (default 64): search effort during index building.
  Higher = better quality graph but slower build.
- **ef_search** (set at query time): search effort during queries. This is
  the main knob you turn in the experiments. Higher = finds more neighbors
  = higher recall but slower. Lower = faster but might miss some.

The critical insight: `ef_search` is a **query-time parameter**. You build
the index once, then you can change `ef_search` for each query without
rebuilding anything. This is why the experiment sweeps `ef_search` across
multiple values (10, 25, 50, 100, 200, 400) — each value is a different
point on the speed/accuracy trade-off curve.

### Exact search vs. approximate search

| Property | Exact | HNSW (approximate) |
|---|---|---|
| How it works | Computes distance to every row, sorts, takes top-k | Navigates a graph, visits only a subset of rows |
| Accuracy | Perfect — always finds the true nearest neighbors | May miss a few neighbors |
| Speed | O(n) — linear in dataset size | Sub-linear — nearly constant for large datasets |
| SQL form | `ORDER BY l2_distance(embedding, query) LIMIT k` | `ORDER BY embedding <-> query LIMIT k` |

### The function-vs-operator trick

This is the single most important technical detail in the project. The
agent built it correctly, but you need to understand *why*.

pgvector gives you two ways to express the same L2 distance:

1. **Function form**: `l2_distance(embedding, $1)` — the planner **cannot**
   use an index to satisfy this. It is forced to do a sequential scan
   (check every row). This guarantees the baseline is truly exhaustive.

2. **Operator form**: `embedding <-> $1` — the planner **can** use the
   HNSW index to satisfy this. This is what makes the approximate search
   fast.

If the exact baseline used the operator form, PostgreSQL might decide to
use the HNSW index for it too — and then your "baseline" would silently
become approximate. Using the function form prevents this by construction.

This is verified in every run with `EXPLAIN (ANALYZE)`: the exact plan
shows "Seq Scan" and the HNSW plan shows "Index Scan using
fv_..._embedding_hnsw_idx".

### Recall@K

A retrieval quality metric. It answers: "of the k true nearest neighbors
that the exact search would find, how many did the approximate search
also find?"

Formula: `Recall@K = |retrieved_neighbors ∩ exact_neighbors| / k`

If Recall@5 = 0.99, it means that on average, 4.95 out of 5 true neighbors
were found by HNSW. A Recall@5 of 1.0 means perfect retrieval.

### Accuracy vs. Recall@K — why they are different

This is subtle but important:

- **Recall@K** measures whether the *same neighbor IDs* were found. If HNSW
  finds neighbor #3 but misses neighbor #7 and finds neighbor #12 instead,
  Recall@K drops — even if neighbor #12 has the same label as neighbor #7.

- **Accuracy** measures whether the *predicted class label* is correct. If
  the missed neighbor and the replacement neighbor have the same label,
  the majority vote gives the same answer and accuracy is unchanged.

This is why you see Recall@5 = 0.9854 but accuracy only drops by 0.17
percentage points — the missed neighbors are usually from the same class
as the ones that replaced them, so the vote is unaffected.

### Tie-aware Recall@K

When two training vectors have exactly the same distance to the query
(they are "tied"), the exact search picks one arbitrarily (by ID). If HNSW
picks the other one, conventional Recall@K counts it as a miss — but it is
actually a valid answer. Tie-aware Recall@K counts any neighbor within a
tiny tolerance of the k-th distance as valid.

On this dataset, there are zero ties, so both metrics are identical. But
the project reports both because in general they measure different things.

### Majority voting

The classification rule. Given k neighbors, each with a label, the
predicted label is the one that appears most often. If there is a tie in
the vote count, the label with the smaller numeric value wins (deterministic).

Optional: distance-weighted voting, where each neighbor votes with weight
`1 / (distance + epsilon)`. Closer neighbors count more. Not used in the
primary experiments.

### Stratified split / stratified sampling

A split where the class proportions are preserved. If the original data is
36% class A and 64% class B, the train and test splits will each also be
roughly 36/64. This matters because if one split is missing a class
entirely, the classifier cannot learn it.

### Data leakage

If you compute statistics (like the mean and standard deviation for
scaling) on the *entire* dataset before splitting, information from the
test set has "leaked" into the training process. The model has seen
information it should not have seen, and the evaluation is optimistic.

This project prevents leakage by fitting the scaler and encoders on the
training split only, then applying them unchanged to the test split. This
is enforced by a test (`test_leakage_contract`).

### Standardization

Subtracting the mean and dividing by the standard deviation, so each
feature has mean=0 and std=1. This matters for KNN because features with
larger numeric ranges would dominate the distance calculation otherwise.
If feature A ranges 0–1000 and feature B ranges 0–1, the L2 distance is
almost entirely determined by feature A.

### EXPLAIN (ANALYZE)

A PostgreSQL command that shows the *execution plan* for a query — which
access path the planner chose, how long it took, how many rows it
scanned. The project runs this on the first query of every batch to
prove that:
- exact queries really use a sequential scan
- HNSW queries really use the index scan

Without this, you would be claiming "HNSW is faster" but have no proof
that the index was actually used.

### Latency percentiles (p50, p95, p99)

- **p50 (median)**: half of the queries are faster than this, half are
  slower.
- **p95**: 95% of queries are faster than this. The slowest 5% are worse.
- **p99**: 99% of queries are faster than this. The slowest 1% are worse.

Percentiles are more informative than the average because they show the
*distribution* of latency, not just the center. A system with a great
average but terrible p99 is unreliable.

### Speed-up

`speed_up = exact_latency / hnsw_latency`

If exact takes 11ms and HNSW takes 0.46ms, the speed-up is 23.8x. This is
the headline number of the project.

---

## Part 3: How the Code Is Organized

### Backend (Python + FastAPI)

The backend lives in `backend/app/`. Here is what each file does:

#### `services/metrics.py` (40 lines — start here)

Defines the `DistanceMetric` enum with three options: L2, cosine, L1. Each
metric has two forms:
- `operator` — the SQL operator (e.g., `<->` for L2) — used by HNSW
- `function` — the SQL function (e.g., `l2_distance`) — used by exact

This tiny file is the bridge between "what distance do we want" and "how
does pgvector express it."

#### `services/classification.py` (260 lines — the ML core)

Contains:
- `majority_vote(labels, distances, weighted)` — the KNN classification
  rule. Takes the labels of the k retrieved neighbors and returns the
  majority. Simple.
- `evaluate_predictions(y_true, y_pred, class_labels)` — computes
  accuracy, macro precision/recall/F1, confusion matrix. This is standard
  ML evaluation.
- `recall_at_k(retrieved, ground_truth)` — conventional Recall@K.
- `tie_aware_recall_at_k(retrieved, candidate_sets)` — tie-tolerant
  version.
- `latency_stats(latencies_ms)` — avg, p50, p95, p99, min, max.
- `repetition_summary(values)` — mean, median, SD, confidence interval
  across repetitions.

**This is the file where the actual machine learning lives.** If you
understand this file, you understand the ML part of the project.

#### `services/preprocessing.py` (477 lines — data preparation)

Contains:
- `fit_preprocessor(df, target_column, mode)` — learns all statistics
  (medians, modes, scaler parameters, one-hot vocabularies) from the
  training split only. Returns a `PreprocessState` and a `Preprocessor`.
- `Preprocessor.transform(df)` — applies the fitted state to new data.
  Never learns anything.
- `validate_dataset(df)` — checks for empty frames, missing labels, single
  class, infinite values.
- `infer_column_types(df)` — classifies each feature as numeric, binary, or
  categorical.
- `vector_literal(matrix)` — converts a numpy matrix into pgvector text
  format (e.g., `[1.2,3.4,5.6]`).

The two preprocessing modes:
- `covertype` (primary): standardize the 10 continuous features, leave the
  44 binary indicators as 0/1. This is correct because the indicators are
  already dimensionless — standardizing them would distort their geometry
  without adding information.
- `standardize_all`: standardize everything, including binary indicators.
  Provided as a comparison option, not the primary setup.

#### `services/vector_store.py` (937 lines — the retrieval engine)

This is the biggest and most important file. It contains:

**Table management:**
- `create_dataset_tables(dataset_id, dimension)` — creates two tables per
  dataset: `fv_<id>` (training pool) and `tv_<id>` (test pool), each with
  a `vector(dim)` column.
- `insert_vectors(dataset_id, which, embeddings, labels, row_indices)` —
  bulk inserts vectors using PostgreSQL's COPY command for speed.

**Why one table per dataset?**
pgvector's HNSW index requires a fixed dimension (`vector(54)`, not bare
`vector`). Also, pgvector applies WHERE filters *after* the index scan, so
a shared table with a `dataset_id` column would return neighbors from the
wrong dataset. One table per dataset avoids both problems.

**Index management:**
- `ensure_hnsw_index(dataset_id, metric, m, ef_construction)` — builds the
  HNSW index, or reuses an existing one if the parameters match. Returns
  an `IndexInfo` describing what happened (built or reused, build time,
  size, parameters).
- `drop_hnsw_index(dataset_id)` — removes the index.
- `hnsw_index_exists(dataset_id)` — checks if the index physically exists.

**Retrieval (the core):**
- `retrieve(query_vector, k, metric, dataset_id, method, ef_search)` —
  runs a single nearest-neighbor query. This is the heart of the project.
  - If method="exact": uses the function form → sequential scan
  - If method="hnsw": uses the operator form → index scan
  - Sets `ef_search` at session level (not `SET LOCAL` because autocommit
    silently discards it), reads it back to confirm, resets it in a
    `finally` block.
  - Runs a warm-up query first (not timed), then the timed query.
  - Captures `EXPLAIN (ANALYZE)` to verify the access path.
- `retrieve_batch(query_vectors, k, metric, dataset_id, method, ef_search,
  repetitions)` — runs the full query set `repetitions` times. One warm-up
  query before the first timed query. Every latency is stored.

**Ground truth:**
- `ground_truth_candidates(query_vectors, k, metric, dataset_id,
  tolerance)` — runs the exhaustive query for every test vector, returns
  the exact top-k IDs plus every ID tied at the k-th distance. This is the
  reference that Recall@K is computed against.

#### `services/experiments.py` (740 lines — the orchestrator)

`run_experiment(config)` is the function that ties everything together:

1. Loads the dataset and preprocessing state from the database.
2. Selects the query vectors from the test pool (stratified sampling).
3. Builds or reuses the HNSW index (for HNSW runs).
4. Computes the exact ground truth (cached across configurations).
5. Runs the retrieval method (exact or HNSW) for `repetitions` passes.
6. For each repetition: classifies via majority vote, computes accuracy /
   Recall@K / latency.
7. Summarizes across repetitions (mean, SD, confidence interval).
8. Persists everything: one `experiments` row, N `experiment_repetitions`
   rows, N*M `query_latencies` rows.
9. Records timing for every phase (index prep, ground truth, retrieval,
   metrics, persistence).

The `ExperimentConfig` dataclass holds all parameters: dataset_id, k,
metric, method, ef_search, repetitions, seeds, benchmark round info, etc.

#### `services/datasets.py` — dataset ingestion

Loads the Covertype CSV, splits it, preprocesses it, and stores the
vectors in pgvector tables.

#### `services/benchmark_order.py` — balanced execution ordering

Generates seeded permutations of the configuration matrix so that every
configuration gets a fair mix of early and late positions across rounds.
This prevents a systematic bias (e.g., exact always running first when the
cache is cold).

#### `api/` — HTTP endpoints

FastAPI routes that expose the functionality as a REST API:
- `datasets.py` — upload/list/delete datasets
- `experiments.py` — run/list/compare experiments
- `health.py` — health check (reports pgvector version, database status)

#### `scripts/run_experiments.py` — the batch runner

Runs the full experiment matrix from the command line. This is what
produced the final results in `docs/results/`. Takes parameters like
`--dataset-id`, `--sizes`, `--ef-search`, `--repetitions`, `--rounds`,
`--seed`, etc.

#### `scripts/verify_results.py` — artifact validator

Reads the results back from the database and CSV files and checks that
everything matches: row counts, index scan flags, applied ef_search
values, etc. Reports 0 errors on the final run.

### Frontend (Next.js + TypeScript)

A web UI for:
- Uploading datasets and viewing their statistics
- Configuring and running experiments
- Viewing results (accuracy, recall, latency charts)
- Comparing experiments side by side
- Reading the methodology

Bilingual (English / Persian) with RTL support. Built with Next.js 16,
TypeScript, Tailwind CSS v4, Recharts for charts, next-intl for i18n.

### Database schema

The key tables:
- `datasets` — metadata about each loaded dataset (name, dimensions,
  preprocessing state, fingerprint)
- `fv_<dataset_id>` — training feature vectors (one table per dataset)
- `tv_<dataset_id>` — test feature vectors (one table per dataset)
- `experiments` — one row per experiment run (71 columns of metrics,
  timing, metadata)
- `experiment_repetitions` — one row per (experiment, repetition)
- `query_latencies` — one row per measured query latency
- `hnsw_indexes` — registry of built HNSW indexes

---

## Part 4: The Experiment Design

### What is being compared?

Two methods for retrieving the k nearest neighbors:

1. **Exact** (baseline): exhaustive sequential scan. Always finds the true
   neighbors. Always correct. Slow for large datasets.

2. **HNSW** (approximate): graph-based index search. Fast. May miss a few
   neighbors. Controlled by `ef_search`.

Both methods feed their retrieved neighbors into the *same* majority vote
function. So any difference in classification accuracy is attributable
*only* to the retrieval stage — the classifier is identical.

### What variables are swept?

| Variable | Values |
|---|---|
| Training pool size | 10,000 and 50,000 rows |
| Method | exact and HNSW |
| ef_search (HNSW only) | 10, 25, 50, 100, 200, 400 |
| k | 5 (fixed in the primary sweep) |
| Distance metric | L2 (fixed in the primary sweep) |
| Queries | all 3,000 held-out rows |
| Repetitions | 3 timed passes |
| Benchmark rounds | 3 seeded balanced rounds |

Total: 2 sizes x 7 configurations (1 exact + 6 HNSW) x 3 rounds = 42
experiments. Each experiment runs 3,000 queries x 3 repetitions = 9,000
latency observations. Total: 42 x 9,000 = 378,000 latency observations.

### What is measured?

For each experiment:
- **Classification quality**: accuracy, macro precision/recall/F1,
  confusion matrix, per-class metrics
- **Retrieval quality**: conventional Recall@K, tie-aware Recall@K
- **Latency**: avg, p50, p95, p99, min, max (in milliseconds)
- **Index cost**: build time, size, reused or fresh
- **Timing breakdown**: time spent on each phase (index prep, ground
  truth, retrieval, metrics, persistence)
- **Plan verification**: did the planner use the index? did it use a
  sequential scan? what ef_search was actually applied?

### What are the three levels of repetition?

This is a subtle but important design point. There are three distinct
levels, and they must not be confused:

1. **Individual query latency** — one timing for one query. 3,000 per
   experiment. These are raw measurements, not population samples.

2. **Repetition** — one complete pass over all 3,000 queries. 3 per
   experiment. These measure run-to-run timing variability (scheduler,
   buffer cache state), not sampling error. The queries and order are
   identical in every repetition.

3. **Benchmark round** — one permutation of the configuration order. 3
   per pool size. These ensure that no configuration is systematically
   advantaged by always running first (when the cache is cold) or last
   (when everything is warm). This is about fairness, not statistics.

---

## Part 5: The Results (and What They Mean)

### Headline numbers (50,000-row training pool)

| Method | ef_search | Recall@5 | Accuracy | Macro F1 | p50 (ms) | Speed-up |
|---|---|---|---|---|---|---|
| Exact | — | 1.0000 | 0.8440 | 0.7500 | 11.063 | — |
| HNSW | 10 | 0.9854 | 0.8423 | 0.7470 | 0.246 | 45.1x |
| HNSW | 25 | 0.9942 | 0.8443 | 0.7499 | 0.280 | 39.6x |
| HNSW | 50 | 0.9974 | 0.8437 | 0.7499 | 0.363 | 30.6x |
| HNSW | 100 | 0.9989 | 0.8440 | 0.7500 | 0.465 | 23.8x |
| HNSW | 200 | 0.9997 | 0.8440 | 0.7500 | 0.753 | 14.7x |
| HNSW | 400 | 0.9998 | 0.8440 | 0.7500 | 1.224 | 9.1x |

### How to read this table

**Read it row by row, left to right:**

- The **exact** row is the baseline: perfect recall (1.0), accuracy 84.4%,
  but takes 11ms per query.
- **HNSW with ef_search=10**: 45x faster (0.25ms vs 11ms), but recall
  drops to 98.5% and accuracy drops by 0.17 percentage points (84.23% vs
  84.40%). Is this acceptable? For most applications, yes — you trade a
  tiny accuracy drop for a 45x speed-up.
- **HNSW with ef_search=100**: 23.8x faster, recall 99.89%, accuracy is
  *identical* to exact (84.40%). This is the sweet spot — you get almost
  perfect retrieval at a large speed-up with *zero* accuracy loss.
- **HNSW with ef_search=400**: 9.1x faster, recall 99.98%, accuracy
  identical. You are spending more time for diminishing returns — ef=100
  is a better deal.

**The story**: As ef_search increases, recall approaches 1.0 and accuracy
matches exact. But latency also increases. There is a diminishing-returns
curve where going from ef=10 to ef=100 buys a lot of recall (98.5% →
99.9%), but going from ef=100 to ef=400 buys almost nothing (99.9% →
99.98%) while doubling the latency.

### The scaling story

| Pool size | Exact p50 | HNSW ef=100 p50 | Speed-up |
|---|---|---|---|
| 10,000 | 2.823 ms | 0.416 ms | 6.8x |
| 50,000 | 11.063 ms | 0.465 ms | 23.8x |

Exact latency grows roughly linearly (5x more data = ~4x slower). HNSW
latency barely changes (0.416 → 0.465ms). So the speed-up *widens* with
dataset size — the larger the dataset, the more you benefit from the
approximate index. This is the core finding.

---

## Part 6: Key Files to Read (in order)

If you want to understand the project, read these files in this order:

1. **`backend/app/services/metrics.py`** (40 lines) — the distance metric
   definitions. Trivial but foundational.

2. **`backend/app/services/classification.py`** (260 lines) — the ML core.
   Majority vote, evaluation metrics, Recall@K. If you understand this,
   you understand 60% of the ML.

3. **`backend/app/services/preprocessing.py`** (477 lines) — how data is
   prepared. Focus on the fit/transform separation and why leakage is
   prevented.

4. **`backend/app/services/vector_store.py`** (937 lines) — the retrieval
   engine. Focus on:
   - The `retrieve()` function (line ~599) — how a single query works
   - The `_distance_expr()` function (line ~536) — the function/operator
     trick
   - The `ensure_hnsw_index()` function (line ~322) — index build/reuse
   - The `ground_truth_candidates()` function (line ~856) — how the
     reference is computed

5. **`backend/app/services/experiments.py`** (740 lines) — how everything
   is orchestrated. Focus on `run_experiment()` (line ~163) — the full
   pipeline from config to persisted results.

6. **`docs/architecture.md`** — explains every design decision and the
   traps it avoids. This is crucial for defending the project.

7. **`docs/experiment-plan.md`** — the experimental design, measurement
   contract, and threats to validity.

---

## Part 7: Questions Your Professor Might Ask

**Q: Why use PostgreSQL instead of a dedicated vector database like
Pinecone or Milvus?**
A: PostgreSQL with pgvector is a general-purpose database that already
handles relational data, transactions, and SQL queries. Adding vector
search to it means you do not need a separate infrastructure. The research
question is specifically about whether this approach is viable, not about
which database is fastest.

**Q: Why not use IVFFlat instead of HNSW?**
A: HNSW generally has better recall-latency trade-offs than IVFFlat. The
project originally considered both but scoped down to HNSW for focus.
This is documented in the decision log.

**Q: How do you know the HNSW index was actually used?**
A: Every query batch runs `EXPLAIN (ANALYZE)` on the first query. The
plan text is checked for "Index Scan using fv_..._embedding_hnsw_idx". If
it is not found, the run is flagged. In the final benchmark, all 36 HNSW
runs verified the index scan; all 6 exact runs verified the sequential
scan.

**Q: Why are repetitions not independent samples?**
A: They reuse the same queries, the same index, and the same order. They
measure run-to-run timing variability (scheduler, cache state), not
sampling error over a population. This is stated explicitly in the docs.

**Q: What are the limitations?**
A: Single dataset (Covertype), single metric (L2) in the full sweep, warm
cache only, single machine, per-query round-trip latency (not batched).
The speed-up numbers are specific to this setup and are not claimed as
universal laws.

**Q: What does ef_search actually do?**
A: It controls how many nodes HNSW visits during the search phase. Higher
values explore more of the graph, finding more candidates and improving
recall, at the cost of more computation. It is set at query time — the
index is built once and reused across all ef_search values.
