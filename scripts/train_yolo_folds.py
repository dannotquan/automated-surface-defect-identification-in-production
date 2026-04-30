#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Optionally train a YOLO model across the four NEU folds.")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model name or path, such as yolov8n.pt.")
    parser.add_argument("--data-root", type=Path, default=Path("NEU-DET"), help="Prepared NEU dataset root.")
    parser.add_argument("--config-dir", type=Path, default=Path("data"), help="Directory containing NEU-DET-<fold>.yaml.")
    parser.add_argument("--folds", type=int, default=4, help="Number of folds to train.")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs per fold.")
    parser.add_argument("--imgsz", type=int, default=200, help="Training image size.")
    parser.add_argument("--project", type=Path, default=Path("runs"), help="Ultralytics output project directory.")
    parser.add_argument("--name-prefix", default="neu-yolo", help="Run name prefix.")
    parser.add_argument("--device", default=None, help="Optional Ultralytics device argument, such as 0 or cpu.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Install optional YOLO dependencies first: python -m pip install -e '.[yolo]'") from exc

    if not args.data_root.exists():
        raise SystemExit(f"Data root not found: {args.data_root}. Run scripts/prepare_neu.py first.")

    for fold in range(1, args.folds + 1):
        data_yaml = args.config_dir / f"NEU-DET-{fold}.yaml"
        if not data_yaml.exists():
            raise SystemExit(f"Missing data config for fold {fold}: {data_yaml}")

        model = YOLO(args.model)
        train_kwargs = {
            "data": str(data_yaml),
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "project": str(args.project),
            "name": f"{args.name_prefix}-fold{fold}",
            "exist_ok": True,
        }
        if args.device is not None:
            train_kwargs["device"] = args.device

        print(f"Training fold {fold} with {args.model} and {data_yaml}")
        model.train(**train_kwargs)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

