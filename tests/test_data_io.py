from pathlib import Path

from surface_defect_benchmark.analysis import load_inputs
from surface_defect_benchmark.data_io import validate_model_table


def test_static_inputs_load_from_repo_data():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    split, folds, models, pvalues = load_inputs(data_dir)

    assert len(split) == 3
    assert len(folds) == 4
    assert "YOLOv5m" in set(models["model"])
    assert pvalues["p_value"].max() < 0.05


def test_model_means_match_fold_scores():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    _, _, models, _ = load_inputs(data_dir)
    validate_model_table(models)

