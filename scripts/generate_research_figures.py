"""Generate Phase 4 research figures and tables from validated Phase 3 outputs.

Inputs are limited to:
  docs/results/validated/summary.csv
  docs/results/validated/statistical_summary.json

Outputs go to docs/results/figures/ and docs/results/tables/.
Raw Phase 3 artifacts are never read or modified.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VALIDATED = ROOT / "docs" / "results" / "validated"
FIG = ROOT / "docs" / "results" / "figures"
TAB = ROOT / "docs" / "results" / "tables"

SUMMARY_CSV = VALIDATED / "summary.csv"
STAT_JSON = VALIDATED / "statistical_summary.json"

NATURAL_COLOR = "#1f77b4"
CONTROLLED_COLOR = "#d62728"
EXACT_COLOR = "#2ca02c"
GRID = "#d9d9d9"


def label(row: pd.Series) -> str:
    if row["retrieval_method"] == "exact":
        base = "Exact"
    else:
        base = f"HNSW ef={int(row['ef_search'])}"
    if row["planner_mode"] == "controlled_hnsw":
        return base + " (controlled)"
    return base


def short_label(size: int, method: str, ef: float) -> str:
    if method == "exact":
        return f"Exact\n{size // 1000}k"
    return f"HNSW ef={int(ef)}\n{size // 1000}k"


def compact_label(row: pd.Series) -> str:
    size = int(row["train_size"]) // 1000
    if row["retrieval_method"] == "exact":
        return f"{size}k Exact"
    suffix = " (ctrl)" if row["planner_mode"] == "controlled_hnsw" else ""
    return f"{size}k H{int(row['ef_search'])}{suffix}"


def finish(fig, caption: str, path: Path) -> None:
    fig.text(0.01, 0.012, caption, fontsize=8, color="#555555", ha="left", va="bottom")
    fig.tight_layout(rect=(0.01, 0.055, 0.99, 0.94))
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def style_axes(ax) -> None:
    ax.grid(axis="both", color=GRID, alpha=0.45, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(SUMMARY_CSV)
    stats = json.loads(STAT_JSON.read_text(encoding="utf-8"))
    natural = df[df["planner_mode"] == "natural"].copy()
    controlled = df[df["planner_mode"] == "controlled_hnsw"].copy()

    natural["config"] = natural.apply(label, axis=1)
    controlled["config"] = controlled.apply(label, axis=1)

    agg = (
        natural.groupby(["train_size", "retrieval_method", "ef_search"], dropna=False)
        .agg(
            runs=("experiment_id", "count"),
            recall_at_k=("recall_at_k", "mean"),
            neighbour_jaccard=("neighbour_jaccard", "mean"),
            accuracy=("accuracy", "mean"),
            precision_macro=("precision_macro", "mean"),
            recall_macro=("recall_macro", "mean"),
            macro_f1=("macro_f1", "mean"),
            latency_p50_ms=("latency_p50_ms", "mean"),
            latency_p95_ms=("latency_p95_ms", "mean"),
            latency_p99_ms=("latency_p99_ms", "mean"),
            latency_min_ms=("latency_min_ms", "mean"),
            latency_max_ms=("latency_max_ms", "mean"),
            index_build_ms=("index_build_ms", "mean"),
            index_size_bytes=("index_size_bytes", "max"),
            speedup_p50=("speedup_p50", "mean"),
        )
        .reset_index()
        .sort_values(["train_size", "retrieval_method", "ef_search"], na_position="first")
    )
    agg["config"] = agg.apply(
        lambda r: "Exact" if r["retrieval_method"] == "exact" else f"HNSW ef={int(r['ef_search'])}",
        axis=1,
    )
    agg["size_label"] = (agg["train_size"] // 1000).astype(int).astype(str) + "k"

    # Table: main benchmark comparison (natural mode only; no failed rows in validated input).
    main_cols = [
        "train_size", "config", "runs", "recall_at_k", "neighbour_jaccard",
        "accuracy", "precision_macro", "recall_macro", "macro_f1",
        "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "latency_min_ms",
        "latency_max_ms", "speedup_p50", "index_build_ms", "index_size_bytes",
    ]
    agg[main_cols].to_csv(TAB / "main_benchmark_comparison.csv", index=False)

    # Table: per-configuration metrics, including controlled mode, with provenance.
    df.to_csv(TAB / "per_configuration_metrics.csv", index=False)

    # Table: statistical comparison.
    stat_rows = []
    for comp in stats["comparisons"]:
        stat_rows.append({
            "train_size": comp["size"],
            "ef_search": comp["ef_search"],
            "n_queries": comp["n_queries"],
            "passes_per_query": comp["passes_per_query"],
            "median_latency_difference_ms": comp["latency"]["median_difference_ms"],
            "median_latency_ci95_low_ms": comp["latency"]["median_difference_ci95_ms"][0],
            "median_latency_ci95_high_ms": comp["latency"]["median_difference_ci95_ms"][1],
            "wilcoxon_p": comp["latency"]["wilcoxon"]["p_two_sided"],
            "wilcoxon_p_holm": comp["latency"]["wilcoxon"]["p_holm"],
            "rank_biserial": comp["latency"]["wilcoxon"]["rank_biserial"],
            "mean_recall_difference": comp["recall_at_5"]["mean_difference"],
            "recall_ci95_low": comp["recall_at_5"]["mean_difference_ci95"][0],
            "recall_ci95_high": comp["recall_at_5"]["mean_difference_ci95"][1],
            "accuracy_difference": comp["classification"]["accuracy_difference"],
            "accuracy_ci95_low": comp["classification"]["accuracy_difference_ci95"][0],
            "accuracy_ci95_high": comp["classification"]["accuracy_difference_ci95"][1],
            "mcnemar_p": comp["classification"]["mcnemar"]["p_two_sided"],
            "mcnemar_p_holm": comp["classification"]["mcnemar"]["p_holm"],
        })
    pd.DataFrame(stat_rows).to_csv(TAB / "statistical_comparison.csv", index=False)

    # Table: index construction and storage.
    idx = agg[agg["retrieval_method"] == "hnsw"][
        ["train_size", "config", "index_build_ms", "index_size_bytes"]
    ].copy()
    idx["index_size_mb"] = (idx["index_size_bytes"] / 1024**2).round(3)
    idx["build_note"] = np.where(
        idx["index_build_ms"].notna(),
        "fresh build on first HNSW run",
        "index reused; build time not paid again",
    )
    idx[["train_size", "config", "index_build_ms", "index_size_bytes", "index_size_mb", "build_note"]].to_csv(
        TAB / "index_construction_storage.csv", index=False
    )

    # Table: per-class availability. Validated outputs contain macro/per-run but not per-class rows.
    pd.DataFrame([
        {
            "availability": "not_available_in_validated_outputs",
            "reason": "PER-CLASS METRICS AND CONFUSION MATRICES WERE RECOMPUTED DURING PROMPT 6 BUT NOT EXPORTED TO docs/results/validated/",
            "consequence": "No per-class table or confusion-matrix figure is generated; this is an explicit unsupported analysis item.",
        }
    ]).to_csv(TAB / "per_class_results.csv", index=False)

    # Table: research finding summary.
    findings = [
        ("RQ1", "HNSW p50 latency is lower than exact for every valid paired configuration.", "Measured + inferential"),
        ("RQ2", "Recall@5 losses are small at higher ef and largest at ef=10, especially on 50k.", "Measured"),
        ("RQ3", "Recall and accuracy do not move monotonically together; classification differences are not significant after Holm.", "Measured + inferential"),
        ("RQ4", "Within each size, higher ef raises latency while improving or preserving Recall@5.", "Measured"),
        ("RQ5", "Latency advantage of HNSW is larger at 50k than 10k.", "Measured"),
        ("RQ6", "Index size is ~5 MB at 10k and ~25 MB at 50k; build time is paid once.", "Measured"),
        ("RQ7", "Controlled HNSW obtains index scans at 10k/ef=400 where natural planner falls back.", "Measured, separate mode"),
        ("RQ8", "Exact is preferable when deterministic full recall and simplicity matter more than latency.", "Interpretation"),
        ("RQ9", "HNSW is preferable when warm-cache latency dominates and small recall losses are acceptable.", "Interpretation"),
    ]
    pd.DataFrame(findings, columns=["research_question", "finding", "evidence_type"]).to_csv(
        TAB / "summary_of_research_findings.csv", index=False
    )

    # Figure 1: latency percentile profiles.
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    for ax, source, title, color in (
        (axes[0], natural, "Natural planner: validated query latency profiles", NATURAL_COLOR),
        (axes[1], controlled, "Controlled HNSW: validated query latency profiles", CONTROLLED_COLOR),
    ):
        ordered = source.sort_values(["train_size", "retrieval_method", "ef_search"], na_position="first").reset_index(drop=True)
        y = np.arange(len(ordered))
        for yi, (_, row) in enumerate(ordered.iterrows()):
            ax.plot([row["latency_min_ms"], row["latency_max_ms"]], [yi, yi], color=color, alpha=0.55, linewidth=2)
            ax.scatter([row["latency_p50_ms"]], [yi], color=EXACT_COLOR if row["retrieval_method"] == "exact" else color, s=36, zorder=3, label="p50" if yi == 0 else None)
            ax.scatter([row["latency_p95_ms"]], [yi], color="#ff7f0e", marker="^", s=30, zorder=3, label="p95" if yi == 0 else None)
            ax.scatter([row["latency_p99_ms"]], [yi], color="#9467bd", marker="s", s=26, zorder=3, label="p99" if yi == 0 else None)
        ax.set_yticks(y)
        ax.set_yticklabels([compact_label(r) for _, r in ordered.iterrows()], fontsize=7, rotation=0, ha="right")
        ax.set_xlabel("Latency (ms)")
        ax.set_title(title)
        style_axes(ax)
    axes[0].legend(loc="upper right", fontsize=8)
    finish(fig, "Source: validated summary.csv. Lines span validated min–max; markers show p50/p95/p99, not raw distributions.", FIG / "figure_01_latency_profiles.png")

    # Figure 2: Recall@5 vs p50 latency.
    fig, ax = plt.subplots(figsize=(9, 6))
    for mode, source, color, marker in (
        ("natural", natural, NATURAL_COLOR, "o"),
        ("controlled_hnsw", controlled, CONTROLLED_COLOR, "X"),
    ):
        for size, sub in source.groupby("train_size"):
            ax.scatter(sub["latency_p50_ms"], sub["recall_at_k"], color=color, marker=marker, s=70, label=f"{mode} | {size // 1000}k" if mode == "natural" else None)
            for _, row in sub.iterrows():
                ax.annotate(label(row).replace(" (controlled)", ""), (row["latency_p50_ms"], row["recall_at_k"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("p50 latency (ms, log scale)")
    ax.set_ylabel("Recall@5")
    ax.set_xscale("log")
    ax.set_title("Recall@5 versus p50 latency by configuration")
    ax.legend(fontsize=8, loc="lower right")
    style_axes(ax)
    finish(fig, "Source: validated summary.csv. Crosses are controlled HNSW diagnostics and are not pooled with natural results.", FIG / "figure_02_recall_vs_latency.png")

    # Figure 3: Recall@5 vs ef_search.
    fig, ax = plt.subplots(figsize=(9, 6))
    for size, sub in natural[natural["retrieval_method"] == "hnsw"].groupby("train_size"):
        ax.plot(sub["ef_search"], sub["recall_at_k"], marker="o", label=f"Natural {size // 1000}k")
        ax.annotate(f"{size // 1000}k", (sub["ef_search"].iloc[-1], sub["recall_at_k"].iloc[-1]), fontsize=8)
    for size, sub in controlled.groupby("train_size"):
        ax.plot(sub["ef_search"], sub["recall_at_k"], marker="X", linestyle="--", color=CONTROLLED_COLOR, alpha=0.8, label=f"Controlled {size // 1000}k")
    exact_recall = natural[natural["retrieval_method"] == "exact"]["recall_at_k"].mean()
    ax.axhline(exact_recall, color=EXACT_COLOR, linestyle=":", label="Exact reference")
    ax.set_xlabel("ef_search")
    ax.set_ylabel("Recall@5")
    ax.set_title("Recall@5 sensitivity to HNSW search effort")
    ax.legend(fontsize=8)
    style_axes(ax)
    finish(fig, "Source: validated summary.csv. Natural 10k/ef=400 is excluded because it failed the access-path check.", FIG / "figure_03_recall_vs_ef.png")

    # Figure 4: accuracy.
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_acc = agg.pivot_table(index="size_label", columns="config", values="accuracy", observed=True)
    pivot_acc.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Training pool")
    ax.set_title("Classification accuracy by validated configuration")
    ax.legend(title="Configuration", fontsize=8)
    ax.set_ylim(0.70, 0.86)
    style_axes(ax)
    finish(fig, "Source: validated summary.csv, natural planner only. Exact is the same-pool reference.", FIG / "figure_04_accuracy_comparison.png")

    # Figure 5: macro-F1.
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot_f1 = agg.pivot_table(index="size_label", columns="config", values="macro_f1", observed=True)
    pivot_f1.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Macro F1")
    ax.set_xlabel("Training pool")
    ax.set_title("Macro-F1 by validated configuration")
    ax.legend(title="Configuration", fontsize=8)
    ax.set_ylim(0.50, 0.78)
    style_axes(ax)
    finish(fig, "Source: validated summary.csv, natural planner only. Lower macro-F1 than accuracy reflects class imbalance.", FIG / "figure_05_macro_f1_comparison.png")

    # Figure 6: training size vs latency.
    fig, ax = plt.subplots(figsize=(9, 6))
    for config, sub in agg.groupby("config", sort=False):
        sub = sub.sort_values("train_size")
        ax.plot(sub["train_size"], sub["latency_p50_ms"], marker="o", label=config)
    ax.set_xlabel("Training rows")
    ax.set_ylabel("p50 latency (ms, log scale)")
    ax.set_yscale("log")
    ax.set_title("Training-pool size versus p50 query latency")
    ax.legend(fontsize=8)
    style_axes(ax)
    finish(fig, "Source: validated summary.csv, natural planner only. Exact latency grows steeply; HNSW remains comparatively flat.", FIG / "figure_06_training_size_vs_latency.png")

    # Figure 7: index size and build time.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    idx_sorted = idx.sort_values(["train_size", "config"])
    x = np.arange(len(idx_sorted))
    axes[0].bar(x, idx_sorted["index_size_mb"], color=NATURAL_COLOR)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{r.config}\n{int(r.train_size) // 1000}k" for _, r in idx_sorted.iterrows()], fontsize=8, rotation=30, ha="right")
    axes[0].set_ylabel("Index size (MB)")
    axes[0].set_title("HNSW index storage")
    builds = idx_sorted.dropna(subset=["index_build_ms"])
    axes[1].bar(np.arange(len(builds)), builds["index_build_ms"], color="#2ca02c")
    axes[1].set_xticks(np.arange(len(builds)))
    axes[1].set_xticklabels([f"{r.config}\n{int(r.train_size) // 1000}k" for _, r in builds.iterrows()], fontsize=8, rotation=30, ha="right")
    axes[1].set_ylabel("Fresh build time (ms)")
    axes[1].set_title("Index construction cost (first build only)")
    for ax in axes:
        style_axes(ax)
    finish(fig, "Source: validated summary.csv. Reused indexes have no fresh build time in this run.", FIG / "figure_07_index_storage_build.png")

    # Figure 8: macro precision vs macro recall.
    fig, ax = plt.subplots(figsize=(9, 6))
    for config, sub in agg.groupby("config", sort=False):
        ax.plot(sub["recall_macro"], sub["precision_macro"], marker="o", label=config)
        for _, row in sub.iterrows():
            ax.annotate(f"{int(row['train_size']) // 1000}k", (row["recall_macro"], row["precision_macro"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Macro recall")
    ax.set_ylabel("Macro precision")
    ax.set_title("Macro precision–recall by validated configuration")
    ax.legend(fontsize=8, loc="lower right")
    style_axes(ax)
    finish(fig, "Source: validated summary.csv. Per-class and confusion-matrix figures are not available in validated outputs.", FIG / "figure_08_macro_precision_recall.png")

    print(f"generated {len(list(FIG.glob('*.png')))} figures and {len(list(TAB.glob('*.csv')))} tables")


if __name__ == "__main__":
    main()
