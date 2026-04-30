#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


DEFAULT_METRIC_COLUMNS = (
    "metrics/mAP50(B)",
    "metrics/mAP50",
    "map50",
    "AP50",
    "ap50",
)


def numeric(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def best_metric_from_results_csv(path: Path, metric_column: str | None = None) -> tuple[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in {path}")

    columns = [metric_column] if metric_column else list(DEFAULT_METRIC_COLUMNS)
    for column in columns:
        if column and column in rows[0]:
            values = [numeric(row.get(column)) for row in rows]
            clean = [value for value in values if value is not None]
            if clean:
                return column, max(clean)

    raise ValueError(f"Could not find an AP50 metric column in {path}. Available columns: {list(rows[0])}")


def collect_results(runs_dir: Path, metric_column: str | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for results_csv in sorted(runs_dir.rglob("results.csv")):
        column, best_ap50 = best_metric_from_results_csv(results_csv, metric_column=metric_column)
        rows.append(
            {
                "run": str(results_csv.parent),
                "results_csv": str(results_csv),
                "metric_column": column,
                "best_ap50": best_ap50,
            }
        )
    return rows


def write_results(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "results_csv", "metric_column", "best_ap50"])
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate AP50 values from Ultralytics results.csv files.")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs"), help="Directory containing YOLO run outputs.")
    parser.add_argument("--output", type=Path, default=Path("reports/summary/yolo_fold_results.csv"))
    parser.add_argument("--metric-column", default=None, help="Optional explicit metric column name.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = collect_results(args.runs_dir, metric_column=args.metric_column)
    if not rows:
        raise SystemExit(f"No results.csv files found under {args.runs_dir}")

    write_results(rows, args.output)
    print(f"Wrote {len(rows)} run summaries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

