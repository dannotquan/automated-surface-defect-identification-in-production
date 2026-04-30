from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import run_analysis


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="surface-defect-benchmark",
        description="Analyze AP50 surface-defect benchmark results from the project CSV artifacts.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser(
        "analyze",
        help="Generate summary statistics, CSV outputs, and figures from stored AP50 results.",
    )
    analyze.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing static CSV result inputs. Default: data",
    )
    analyze.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports"),
        help="Directory for generated summaries and figures. Default: reports",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        result = run_analysis(data_dir=args.data_dir, output_dir=args.output_dir)
        print("Analysis complete.")
        print(f"Summary tables: {result.summary_dir}")
        print(f"Figures: {result.figure_dir}")
        print(f"Top model by mean AP50: {result.top_model} ({result.top_ap50:.3f})")
        print(f"ANOVA excluding YOLOv8n-DD: p = {result.anova_p_value:.4f}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2
