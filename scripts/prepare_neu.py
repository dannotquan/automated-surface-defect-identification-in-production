#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


CLASS_TO_ID = {
    "crazing": 0,
    "inclusion": 1,
    "patches": 2,
    "pitted_surface": 3,
    "rolled-in_scale": 4,
    "scratches": 5,
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def class_id(name: str) -> int:
    normalized = name.strip()
    if normalized not in CLASS_TO_ID:
        raise ValueError(f"Unknown NEU defect class '{name}'. Expected one of {sorted(CLASS_TO_ID)}")
    return CLASS_TO_ID[normalized]


def voc_box_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: float,
    image_height: float,
) -> tuple[float, float, float, float]:
    if xmax <= xmin or ymax <= ymin:
        raise ValueError(f"Invalid bounding box: {(xmin, ymin, xmax, ymax)}")
    return (
        ((xmax + xmin) / 2.0) / image_width,
        ((ymax + ymin) / 2.0) / image_height,
        (xmax - xmin) / image_width,
        (ymax - ymin) / image_height,
    )


def image_size_from_xml(root: ET.Element, default_width: int = 200, default_height: int = 200) -> tuple[int, int]:
    size = root.find("size")
    if size is None:
        return default_width, default_height

    width = int(float(size.findtext("width", str(default_width))))
    height = int(float(size.findtext("height", str(default_height))))
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size in XML annotation: {(width, height)}")
    return width, height


def convert_xml_annotation(xml_path: Path, label_path: Path) -> None:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_width, image_height = image_size_from_xml(root)
    lines: list[str] = []

    for obj in root.findall("object"):
        name = obj.findtext("name")
        if not name:
            raise ValueError(f"Missing object class name in {xml_path}")
        box = obj.find("bndbox")
        if box is None:
            raise ValueError(f"Missing bndbox in {xml_path}")

        xmin = float(box.findtext("xmin", "nan"))
        ymin = float(box.findtext("ymin", "nan"))
        xmax = float(box.findtext("xmax", "nan"))
        ymax = float(box.findtext("ymax", "nan"))
        x_center, y_center, width, height = voc_box_to_yolo(xmin, ymin, xmax, ymax, image_width, image_height)
        lines.append(f"{class_id(name)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def normalize_kaggle_layout(dataset_dir: Path) -> None:
    all_images = dataset_dir / "all" / "images"
    all_annotations = dataset_dir / "all" / "annotations"
    all_labels = dataset_dir / "all" / "labels"
    all_images.mkdir(parents=True, exist_ok=True)
    all_annotations.mkdir(parents=True, exist_ok=True)
    all_labels.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation"):
        image_root = dataset_dir / split / "images"
        annotation_root = dataset_dir / split / "annotations"
        if image_root.exists():
            for image_path in image_root.rglob("*"):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    shutil.copy2(image_path, all_images / image_path.name)
        if annotation_root.exists():
            for xml_path in annotation_root.rglob("*.xml"):
                shutil.copy2(xml_path, all_annotations / xml_path.name)


def infer_class_from_filename(filename: str) -> str:
    for class_name in CLASS_TO_ID:
        if filename.startswith(class_name):
            return class_name
    raise ValueError(f"Could not infer NEU class from filename '{filename}'")


def create_stratified_folds(image_names: list[str], k: int = 4, seed: int = 244) -> list[list[str]]:
    if k < 2:
        raise ValueError("At least two folds are required for train/test validation.")

    grouped: dict[str, list[str]] = defaultdict(list)
    for name in image_names:
        grouped[infer_class_from_filename(name)].append(name)

    small_classes = {class_name: len(names) for class_name, names in grouped.items() if len(names) < k}
    if small_classes:
        raise ValueError(f"Each class needs at least {k} images. Too small: {small_classes}")

    rng = random.Random(seed)
    folds: list[list[str]] = [[] for _ in range(k)]
    for names in grouped.values():
        names = sorted(names)
        rng.shuffle(names)
        for index, image_name in enumerate(names):
            folds[index % k].append(image_name)

    return [sorted(fold) for fold in folds]


def copy_fold_files(dataset_dir: Path, fold_index: int, test_images: list[str], train_images: list[str]) -> None:
    source_images = dataset_dir / "all" / "images"
    source_labels = dataset_dir / "all" / "labels"

    for split_name, names in ((f"test{fold_index}", test_images), (f"train{fold_index}", train_images)):
        image_target = dataset_dir / split_name / "images"
        label_target = dataset_dir / split_name / "labels"
        image_target.mkdir(parents=True, exist_ok=True)
        label_target.mkdir(parents=True, exist_ok=True)
        for image_name in names:
            label_name = f"{Path(image_name).stem}.txt"
            shutil.copy2(source_images / image_name, image_target / image_name)
            shutil.copy2(source_labels / label_name, label_target / label_name)


def write_yolo_yaml(dataset_dir: Path, config_dir: Path, fold_index: int) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = config_dir / f"NEU-DET-{fold_index}.yaml"
    names = ", ".join(f"{idx}: {name}" for name, idx in CLASS_TO_ID.items())
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_dir.resolve()}",
                f"train: train{fold_index}/images",
                f"val: test{fold_index}/images",
                "names:",
                *[f"  {idx}: {name}" for name, idx in CLASS_TO_ID.items()],
                "",
                f"# Classes: {names}",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def prepare_dataset(dataset_dir: Path, config_dir: Path, folds: int = 4, seed: int = 244) -> list[Path]:
    normalize_kaggle_layout(dataset_dir)

    annotation_dir = dataset_dir / "all" / "annotations"
    label_dir = dataset_dir / "all" / "labels"
    image_dir = dataset_dir / "all" / "images"
    if not annotation_dir.exists() or not image_dir.exists():
        raise FileNotFoundError(
            f"Expected {dataset_dir}/all/images and {dataset_dir}/all/annotations. "
            "Download/unzip the Kaggle NEU dataset first."
        )

    for xml_path in sorted(annotation_dir.glob("*.xml")):
        convert_xml_annotation(xml_path, label_dir / f"{xml_path.stem}.txt")

    image_names = sorted(path.name for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not image_names:
        raise FileNotFoundError(f"No supported images found in {image_dir}. Expected: {IMAGE_EXTENSIONS}")

    stratified_folds = create_stratified_folds(image_names, k=folds, seed=seed)
    yaml_paths: list[Path] = []
    for index, test_images in enumerate(stratified_folds, start=1):
        test_set = set(test_images)
        train_images = [name for name in image_names if name not in test_set]
        copy_fold_files(dataset_dir, index, test_images, train_images)
        yaml_paths.append(write_yolo_yaml(dataset_dir, config_dir, index))

    return yaml_paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare NEU-DET for four-fold YOLO benchmarking.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("NEU-DET"), help="Path to the unzipped NEU-DET folder.")
    parser.add_argument("--config-dir", type=Path, default=Path("data"), help="Directory for generated YOLO YAML files.")
    parser.add_argument("--folds", type=int, default=4, help="Number of stratified folds to create.")
    parser.add_argument("--seed", type=int, default=244, help="Random seed for fold assignment.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    yaml_paths = prepare_dataset(args.dataset_dir, args.config_dir, folds=args.folds, seed=args.seed)
    print("Prepared NEU dataset folds.")
    for path in yaml_paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
