# Project Alignment and Rubric Map

## Relationship to the Proposed Project

This repository is based on the proposed project idea: automated surface defect identification in production using reproducible model benchmarking. The project idea selected Lema et al. (2025) as the main reference because it uses a public surface-defect dataset, compares deep learning detection methods, and applies statistical testing to decide whether model differences are meaningful.

The code repo is not only a summary of `s10845-025-02672-8 copy.pdf`. It uses that paper as the main methodological source, then turns the project into a reproducible artifact with:

- Static AP50 result tables for default analysis without GPU or dataset access.
- Generated plots for train/test split sensitivity, fold variation, model mean AP50, and p-values.
- ANOVA and Tukey HSD analysis code for statistically supported comparison.
- Optional NEU dataset preparation and YOLO training/evaluation scripts.
- Documentation that connects the work to Week 12 computer vision concepts: detection, small datasets, feature hierarchies, augmentation, pretrained models, and fine-tuning.

## Workspace Resources Used

- `Project Idea - Duc Quang.pdf`: defines the project direction and selected paper.
- `Project Proposal ISE 244 - Duc Quang.pdf`: provides problem framing, novelty, broader implications, and limitations.
- `Model Card.pdf`: provides intended use, factors, metrics, ethical considerations, data protocol, and caveats.
- `s10845-025-02672-8 copy.pdf`: provides the core benchmark methodology and reported AP50/statistical results.
- `graph1_ap50_by_split (1) copy.png` through `graph4_pvalues (1) copy.png`: guide the report-supporting generated figures.
- `week12.pdf`: supports the computer vision method framing.

## Canvas Rubric Support

This repository supports the final report rubric as follows:

- Problem Definition: documents the industrial need for automated surface-defect detection and the problem of unfair model comparison.
- Project Objectives: implements a benchmark-supporting artifact for comparing object detection models using AP50, fold stability, and statistical tests.
- Analysis: provides code for loading AP50 data, ranking models, running ANOVA, and running Tukey HSD.
- Results: generates summary CSV files and four figures that can be inserted into the report.
- Discussion: supports the conclusion that AP50 alone can be misleading when dataset partitions and statistical significance are ignored.
- Evaluation and Reflection: documents limitations, including missing local dataset/weights and the difference between static analysis and full retraining.
- Artifacts: provides a submission-ready code repository with data tables, generated figures, optional reproducibility scripts, tests, and README instructions.

## Honest Scope

The default repo path is intentionally lightweight and reproducible from the included CSV files. It does not claim that full YOLO training was rerun locally. Full retraining requires the external NEU dataset, optional Ultralytics installation, and suitable compute.

