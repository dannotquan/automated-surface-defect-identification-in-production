from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def save_split_plot(split: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "graph1_ap50_by_split.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels = [f"{int(train)} / {int(test)}" for train, test in zip(split["train_percent"], split["test_percent"])]
    bars = ax.bar(labels, split["ap50"], color="#377eb8", width=0.55)
    ax.plot(labels, split["ap50"], color="#1f4e79", marker="o", linewidth=2)
    ax.set_title("AP50 by Train/Test Split")
    ax.set_xlabel("Train / Test split")
    ax.set_ylabel("AP50")
    ax.set_ylim(0.70, 0.80)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_fold_plot(folds: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "graph2_ap50_by_fold.png"
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(folds["fold"].astype(str), folds["ap50"], color="#4daf4a", width=0.55)
    mean_ap50 = folds["ap50"].mean()
    ax.axhline(mean_ap50, color="#333333", linestyle="--", linewidth=1.5, label=f"mean {mean_ap50:.3f}")
    ax.set_title("AP50 by Fold - YOLOv8n")
    ax.set_xlabel("Fold")
    ax.set_ylabel("AP50")
    ax.set_ylim(0.73, 0.775)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_model_plot(models: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "graph3_mean_ap50_by_model.png"
    ranked = models.sort_values("mean_ap50", ascending=True)
    colors = ["#e41a1c" if row["model"] == "YOLOv8n-DD" else "#984ea3" for _, row in ranked.iterrows()]

    height = max(6, 0.35 * len(ranked))
    fig, ax = plt.subplots(figsize=(8.5, height))
    bars = ax.barh(ranked["model"], ranked["mean_ap50"], color=colors)
    ax.set_title("Four-Fold Cross-Validation Mean AP50 by Model")
    ax.set_xlabel("Mean AP50")
    ax.set_xlim(0.30, 0.80)
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_pvalue_plot(pvalues: pd.DataFrame, figure_dir: Path) -> Path:
    path = figure_dir / "graph4_pvalues.png"
    labels = pvalues["comparison"].str.replace(" including ", "\nincluding ", regex=False)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, pvalues["p_value"], color="#ff7f00", width=0.55)
    ax.axhline(0.05, color="#333333", linestyle="--", linewidth=1.5, label="p = 0.05")
    ax.set_title("Statistical Significance - p-values")
    ax.set_ylabel("p-value")
    ax.set_ylim(0, max(0.07, float(pvalues["p_value"].max()) * 1.25))
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path

