from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_dir(data_dir: Path) -> Path:
    """Resolve a data directory from CLI input or the source-tree default."""
    if data_dir.exists():
        return data_dir

    source_tree_data = PROJECT_ROOT / "data"
    if source_tree_data.exists():
        return source_tree_data

    raise FileNotFoundError(
        f"Could not find data directory '{data_dir}' or source-tree data directory '{source_tree_data}'."
    )


def read_csv(data_dir: Path, filename: str) -> pd.DataFrame:
    path = resolve_data_dir(data_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Required input CSV not found: {path}")
    return pd.read_csv(path)


def fold_columns() -> list[str]:
    return ["fold_1", "fold_2", "fold_3", "fold_4"]


def validate_model_table(models: pd.DataFrame, tolerance: float = 0.002) -> None:
    missing = {"model", "mean_ap50", *fold_columns()} - set(models.columns)
    if missing:
        raise ValueError(f"model_fold_ap50.csv is missing columns: {sorted(missing)}")

    calculated = models[fold_columns()].mean(axis=1)
    declared = models["mean_ap50"]
    bad = models.loc[(calculated - declared).abs() > tolerance, "model"].tolist()
    if bad:
        raise ValueError(f"Mean AP50 does not match fold scores for: {bad}")
