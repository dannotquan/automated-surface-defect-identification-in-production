from pathlib import Path

from surface_defect_benchmark.analysis import load_inputs
from surface_defect_benchmark.statistics import anova_summary, tukey_hsd_summary


def test_anova_finds_significance_with_defectdef():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    _, _, models, _ = load_inputs(data_dir)
    summary = anova_summary(models)

    all_models = summary.loc[summary["comparison_set"] == "All models including YOLOv8n-DD"].iloc[0]
    assert all_models["p_value"] < 0.05
    assert all_models["eta_squared"] > 0


def test_tukey_summary_contains_key_paper_comparison():
    data_dir = Path(__file__).resolve().parents[1] / "data"
    _, _, models, _ = load_inputs(data_dir)
    tukey = tukey_hsd_summary(models)

    key = tukey[(tukey["model_a"] == "YOLOv5m") & (tukey["model_b"] == "YOLOv10s")]
    assert not key.empty
    assert key.iloc[0]["mean_difference_a_minus_b"] > 0
