# Surface Defect Detection Benchmark

This project studies automated surface defect identification in manufacturing using reproducible benchmarking methods for deep learning object detection models. The focus is not only model accuracy, but also whether reported AP50 differences remain meaningful across train/test partitions and cross-validation folds.

The main reference is Lema, Sanchez-Gonzalez, Usamentiaga, and delaCalle (2025), "Benchmarking deep learning models for surface defect detection: a reproducible and statistically-rigorous approach," Journal of Intelligent Manufacturing.

## Project Objectives

- Analyze surface defect detection as an industrial quality-control problem.
- Compare object detection model results using AP50.
- Examine sensitivity to train/test split selection and fold variation.
- Apply one-way ANOVA and Tukey HSD to evaluate statistical significance.
- Provide a lightweight reproducibility structure for the NEU surface defect dataset.

## Repository Contents

- `surface_defect_benchmark/`: Python package for analysis, statistics, plotting, and command-line execution.
- `data/`: AP50 result tables and p-value summary used for reproducible analysis.
- `reports/summary/`: generated CSV summaries for rankings, ANOVA, and Tukey HSD.
- `reports/figures/`: generated figures for AP50 split sensitivity, fold variation, model means, and p-values.
- `mini_app/`: static browser app for exploring benchmark results through an industrial quality-control dashboard.
- `scripts/`: optional NEU dataset preparation, YOLO fold training, and YOLO run aggregation scripts.
- `tests/`: unit tests for data loading, statistics, annotation conversion, and fold creation.

## Quick Start

From this repository folder:

```bash
python -m pip install -e .
python -m surface_defect_benchmark analyze
```

The analysis command reads CSV files in `data/` and writes:

- `reports/summary/model_rankings.csv`
- `reports/summary/anova_summary.csv`
- `reports/summary/tukey_hsd_summary.csv`
- `reports/figures/graph1_ap50_by_split.png`
- `reports/figures/graph2_ap50_by_fold.png`
- `reports/figures/graph3_mean_ap50_by_model.png`
- `reports/figures/graph4_pvalues.png`

## Mini App

Open `mini_app/index.html` in a browser to interact with the benchmark results. The app illustrates a steel-surface inspection frame, fold-level AP50 scores, model rankings, split sensitivity, and the statistical significance result.

## Optional Dataset Preparation

Download the NEU dataset from Kaggle and unzip it so `NEU-DET/` is inside this repo. The expected source layout is the Kaggle layout with `train/` and `validation/` folders, or a normalized layout with `all/images` and `all/annotations`.

```bash
python scripts/prepare_neu.py --dataset-dir NEU-DET --config-dir data
```

This converts PASCAL VOC XML annotations into YOLO labels, creates four stratified train/test folds, and writes `data/NEU-DET-1.yaml` through `data/NEU-DET-4.yaml`.

## Optional YOLO Training

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

- Dataset: NEU surface defect dataset, 1,800 grayscale steel-surface images.
- Classes: crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches.
- Task type: object detection, not segmentation. Detection predicts bounding boxes and labels; segmentation would require pixel-level masks.
- Main metric: AP50, average precision at IoU 0.50.
- Evaluation design: four-fold stratified cross-validation using 75% training and 25% testing in each fold.
- Statistical analysis: one-way ANOVA to test whether group differences exist, followed by Tukey HSD for model-pair comparisons.
- Effect size: eta-squared is included with ANOVA results to show how much AP50 variation is explained by model choice.

The computer vision framing also follows course concepts on detection, small datasets, feature hierarchies, augmentation, pretrained models, and fine-tuning.

## Testing

Install development tools:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m surface_defect_benchmark analyze
```

## Scope

The default analysis is reproducible from the included CSV files and does not require the NEU image dataset or GPU hardware. Full YOLO training requires the external NEU dataset, optional Ultralytics installation, and suitable compute resources.

## References

- Lema, D. G., Sanchez-Gonzalez, L., Usamentiaga, R., & delaCalle, F. J. (2025). Benchmarking deep learning models for surface defect detection: a reproducible and statistically-rigorous approach. Journal of Intelligent Manufacturing.
- Official paper code: https://github.com/darioglema/Statistical-Significance-DefectDetection
- NEU surface defect dataset: https://www.kaggle.com/datasets/kaustubhdikshit/neu-surface-defect-database
