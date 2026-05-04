from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .data_io import read_csv, validate_model_table
from .plots import save_fold_plot, save_model_plot, save_pvalue_plot, save_split_plot
from .statistics import anova_summary, tukey_hsd_summary


@dataclass(frozen=True)
class AnalysisResult:
    summary_dir: Path
    figure_dir: Path
    top_model: str
    top_ap50: float
    anova_p_value: float
    anova_eta_squared: float


def load_inputs(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = read_csv(data_dir, "split_sensitivity.csv")
    folds = read_csv(data_dir, "yolov8n_fold_scores.csv")
    models = read_csv(data_dir, "model_fold_ap50.csv")
    pvalues = read_csv(data_dir, "pvalue_summary.csv")
    validate_model_table(models)
    return split, folds, models, pvalues


def run_analysis(data_dir: Path = Path("data"), output_dir: Path = Path("reports")) -> AnalysisResult:
    split, folds, models, pvalues = load_inputs(data_dir)

    summary_dir = output_dir / "summary"
    figure_dir = output_dir / "figures"
    summary_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rankings = models.sort_values("mean_ap50", ascending=False).reset_index(drop=True)
    rankings.to_csv(summary_dir / "model_rankings.csv", index=False)

    anova = anova_summary(models)
    anova.to_csv(summary_dir / "anova_summary.csv", index=False)

    tukey = tukey_hsd_summary(models, include_defectdef=False)
    tukey.to_csv(summary_dir / "tukey_hsd_summary.csv", index=False)

    save_split_plot(split, figure_dir)
    save_fold_plot(folds, figure_dir)
    save_model_plot(models, figure_dir)
    save_pvalue_plot(pvalues, figure_dir)

    top = rankings.iloc[0]
    competitive_anova = anova.loc[anova["comparison_set"] == "All models excluding YOLOv8n-DD"].iloc[0]
    return AnalysisResult(
        summary_dir=summary_dir,
        figure_dir=figure_dir,
        top_model=str(top["model"]),
        top_ap50=float(top["mean_ap50"]),
        anova_p_value=float(competitive_anova["p_value"]),
        anova_eta_squared=float(competitive_anova["eta_squared"]),
    )
