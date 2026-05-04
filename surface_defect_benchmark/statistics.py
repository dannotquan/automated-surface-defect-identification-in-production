from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from .data_io import fold_columns


@dataclass(frozen=True)
class AnovaResult:
    label: str
    f_statistic: float
    p_value: float
    n_groups: int
    n_observations: int


def model_groups(models: pd.DataFrame) -> tuple[list[str], list[np.ndarray]]:
    names = models["model"].tolist()
    groups = [row[fold_columns()].to_numpy(dtype=float) for _, row in models.iterrows()]
    return names, groups


def one_way_anova(models: pd.DataFrame, label: str) -> AnovaResult:
    _, groups = model_groups(models)
    f_statistic, p_value = stats.f_oneway(*groups)
    return AnovaResult(
        label=label,
        f_statistic=float(f_statistic),
        p_value=float(p_value),
        n_groups=len(groups),
        n_observations=sum(len(group) for group in groups),
    )


def eta_squared(models: pd.DataFrame) -> float:
    """Return one-way ANOVA eta-squared effect size for fold AP50 groups."""
    _, groups = model_groups(models)
    values = np.concatenate(groups)
    grand_mean = values.mean()
    between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
    total = sum((value - grand_mean) ** 2 for value in values)
    return float(between / total) if total else 0.0


def anova_summary(models: pd.DataFrame) -> pd.DataFrame:
    comparison_sets = [
        ("All models including YOLOv8n-DD", models),
        (
            "All models excluding YOLOv8n-DD",
            models[models["model"] != "YOLOv8n-DD"].reset_index(drop=True),
        ),
    ]
    rows = []
    for label, frame in comparison_sets:
        result = one_way_anova(frame, label)
        rows.append(
            {
                "comparison_set": result.label,
                "f_statistic": result.f_statistic,
                "p_value": result.p_value,
                "eta_squared": eta_squared(frame),
                "n_groups": result.n_groups,
                "n_observations": result.n_observations,
                "significant_at_0_05": result.p_value < 0.05,
            }
        )

    return pd.DataFrame(rows)


def tukey_hsd_summary(models: pd.DataFrame, include_defectdef: bool = False) -> pd.DataFrame:
    """Run Tukey HSD pairwise comparisons over fold AP50 groups."""
    working = models.copy()
    if not include_defectdef:
        working = working[working["model"] != "YOLOv8n-DD"].reset_index(drop=True)

    names, groups = model_groups(working)
    result = stats.tukey_hsd(*groups)

    rows: list[dict[str, object]] = []
    for i, model_a in enumerate(names):
        for j, model_b in enumerate(names):
            if j <= i:
                continue
            rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "mean_ap50_a": float(working.loc[i, "mean_ap50"]),
                    "mean_ap50_b": float(working.loc[j, "mean_ap50"]),
                    "mean_difference_a_minus_b": float(working.loc[i, "mean_ap50"] - working.loc[j, "mean_ap50"]),
                    "p_value": float(result.pvalue[i, j]),
                    "significant_at_0_05": bool(result.pvalue[i, j] < 0.05),
                }
            )

    return pd.DataFrame(rows).sort_values(["p_value", "model_a", "model_b"]).reset_index(drop=True)
