import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_yolo_folds.py"
SPEC = importlib.util.spec_from_file_location("evaluate_yolo_folds", SCRIPT_PATH)
evaluate_yolo_folds = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate_yolo_folds)


def test_results_csv_parser_handles_ultralytics_spacing():
    csv_text = """epoch, metrics/mAP50(B) , other
1, 0.702 , a
2, 0.754 , b
3, , c
"""
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "results.csv"
        path.write_text(csv_text, encoding="utf-8")

        column, best_ap50 = evaluate_yolo_folds.best_metric_from_results_csv(path)

    assert column == "metrics/mAP50(B)"
    assert best_ap50 == 0.754
