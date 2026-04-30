# Surface Defect Detection Benchmark

This repository supports the ISE 244 final project on automated surface defect detection in manufacturing. It is a lightweight, submission-friendly companion to the report: the default analysis runs from included CSV result tables, while optional scripts show how the NEU dataset and YOLO fold experiments can be prepared when data and GPU resources are available.

Main reference: Lema, Sanchez-Gonzalez, Usamentiaga, and delaCalle (2025), "Benchmarking deep learning models for surface defect detection: a reproducible and statistically-rigorous approach," Journal of Intelligent Manufacturing.

Additional references:

- Official paper code: https://github.com/darioglema/Statistical-Significance-DefectDetection
- NEU dataset: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
- Course support: Week 12 lecture on computer vision, small datasets, feature hierarchies, detection, augmentation, pretrained models, and fine-tuning.

See `PROJECT_ALIGNMENT.md` for how this code repo connects to the original project idea, proposal, model card, and Canvas rubric.

## What This Repo Does

- Recreates the project's AP50 analysis from stored paper/model-card results.
- Generates summary tables and four figures for the final report.
- Runs one-way ANOVA and Tukey HSD on four-fold AP50 scores, including a competitive-model subset that excludes the extreme YOLOv8n-DefectDef underperformer.
- Provides optional scripts for NEU XML-to-YOLO conversion, stratified fold setup, YOLO fold training, and run aggregation.

This repo does not include NEU images, Kaggle files, trained weights, or claims that full training was rerun locally. Those files are large or externally licensed and should be downloaded separately if needed.

## Quick Start: Report-Supporting Analysis

From this repository folder:

```bash
python -m pip install -e .
python -m surface_defect_benchmark analyze
```

The command reads CSV files in `data/` and writes:

- `reports/summary/model_rankings.csv`
- `reports/summary/anova_summary.csv`
- `reports/summary/tukey_hsd_summary.csv`
- `reports/figures/graph1_ap50_by_split.png`
- `reports/figures/graph2_ap50_by_fold.png`
- `reports/figures/graph3_mean_ap50_by_model.png`
- `reports/figures/graph4_pvalues.png`

## Optional: Prepare the NEU Dataset

Download the NEU dataset from Kaggle and unzip it so `NEU-DET/` is inside this repo. The expected source layout is the Kaggle layout with `train/` and `validation/` folders, or a normalized layout with `all/images` and `all/annotations`.

```bash
python scripts/prepare_neu.py --dataset-dir NEU-DET --config-dir data
```

This converts PASCAL VOC XML annotations into YOLO labels, creates four stratified train/test folds, and writes `data/NEU-DET-1.yaml` through `data/NEU-DET-4.yaml`.

## Optional: Train and Evaluate YOLO Folds

Install the heavier optional dependency first:

```bash
python -m pip install -e ".[yolo]"
```

Train one model across the four folds:

```bash
python scripts/train_yolo_folds.py --model yolov8n.pt --data-root NEU-DET --config-dir data --epochs 50
```

Aggregate resulting AP50 values:

```bash
python scripts/evaluate_yolo_folds.py --runs-dir runs --output reports/summary/yolo_fold_results.csv
```

## Method Summary

The benchmark follows the paper's core methodology:

- Dataset: NEU surface defect dataset, 1,800 grayscale steel-surface images.
- Classes: crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches.
- Task type: object detection, not segmentation. Detection predicts bounding boxes and labels; segmentation would require pixel-level masks.
- Main metric: AP50, average precision at IoU 0.50.
- Evaluation design: four-fold stratified cross-validation using 75% training and 25% testing in each fold.
- Statistical analysis: one-way ANOVA to test whether group differences exist, followed by Tukey HSD for model-pair comparisons.

Week 12 lecture ideas appear in the documentation and comments: small industrial datasets benefit from careful validation, data augmentation, pretrained feature extractors, and fine-tuning rather than relying on a single lucky train/test split.

## Testing

Install development tools:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m surface_defect_benchmark analyze
```

## Submission Notes

For Canvas submission, include this repository folder with the report. The analysis outputs are reproducible from the included CSV files. The optional YOLO scripts are provided as transparent reproducibility support and require the external NEU dataset plus additional compute.

The repo is a supporting artifact, not the entire report by itself. The final report should use this repository as evidence for the Analysis, Results, Discussion, Evaluation and Reflection, and Artifacts sections.
