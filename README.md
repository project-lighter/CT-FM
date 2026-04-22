# CT-FM Feature Extractor — CVPR 2026 CT FM Challenge

This repo contains the full pipeline for the [CVPR 2026 3D CT Foundation Model Challenge](https://github.com/kmin940/CVPR26-3DCTFMCompetition): feature extraction with the CT-FM model (SegResNet backbone) and linear probing evaluation across multiple diseases on AMOS, COVID-CT, and LUNA25 datasets.

## Overview

The pipeline has three stages:

1. **Feature extraction** — runs CT-FM on a directory of `.nii.gz` CT scans and writes 512-dim embeddings to `.h5` files (runs inside Docker)
2. **Linear probing** — trains multi-head linear classifiers on those embeddings against disease labels, sweeping 13 learning rates (runs outside Docker)
3. **Prediction** — generates per-scan predictions CSV with logits, probabilities, and GT labels

## Results

### AMOS (abdominal, HU window: -1024 to 2048)

| Disease | Balanced Acc | AUROC |
|---|---|---|
| fatty_liver | 0.833 | 0.869 |
| splenomegaly | 0.814 | 0.840 |
| atherosclerosis | 0.765 | 0.774 |
| adrenal_hyperplasia | 0.757 | 0.725 |
| colorectal_cancer | 0.750 | 0.762 |
| cholecystitis | 0.735 | 0.743 |
| ascites | 0.721 | 0.691 |
| hydronephrosis | 0.719 | 0.725 |
| kidney_stone | 0.665 | 0.655 |
| gallstone | 0.655 | 0.663 |
| liver_lesion | 0.627 | 0.635 |
| lymphadenopathy | 0.625 | 0.651 |
| liver_cyst | 0.615 | 0.633 |
| renal_cyst | 0.609 | 0.600 |
| liver_calcifications | 0.597 | 0.526 |

### COVID-CT (chest, HU window: -1000 to 400)

| Disease | Balanced Acc | AUROC |
|---|---|---|
| covid | 0.656 | 0.636 |

### LUNA25 (chest, HU window: -1000 to 400)

| Disease | Balanced Acc | AUROC |
|---|---|---|
| lung_nodule_malignancy | 0.620 | 0.584 |

## Setup

```bash
pip install -r requirements.txt
```

Weights are downloaded automatically from HuggingFace on first run (`project-lighter/ct_fm_feature_extractor`).

## Stage 1 — Feature Extraction

### Docker

Build the image (weights are baked in at build time):

```bash
docker build -f Dockerfile -t ct_fm_extractor .
```

Run extraction — abdominal (default HU window):

```bash
docker run --gpus all \
  -v /path/to/inputs:/workspace/inputs \
  -v /path/to/outputs:/workspace/outputs \
  ct_fm_extractor bash extract_feat_LP.sh
```

Run extraction — chest CT (lung window):

```bash
docker run --gpus all \
  -v /path/to/inputs:/workspace/inputs \
  -v /path/to/outputs:/workspace/outputs \
  -e HU_MIN=-1000 -e HU_MAX=400 \
  ct_fm_extractor bash extract_feat_LP.sh
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `INPUT_DIR` | `/workspace/inputs` | Directory of `.nii.gz` CT scans |
| `OUTPUT_DIR` | `/workspace/outputs` | Output directory for `.h5` embedding files |
| `MASKS_DIR` | _(unset)_ | Optional directory of segmentation masks for ROI cropping |
| `HU_MIN` | `-1024` | HU window lower bound |
| `HU_MAX` | `2048` | HU window upper bound |

### Without Docker

```bash
python extract_feat_LP.py -i /path/to/inputs -o /path/to/outputs
# Lung window:
python extract_feat_LP.py -i /path/to/inputs -o /path/to/outputs --hu_min -1000 --hu_max 400
```

Each output `.h5` file contains a `y_hat` dataset of shape `(512,)`.

Multi-GPU is supported automatically — extraction is distributed across all available GPUs.

## Stage 2 — Linear Probing

Runs outside Docker. Expects `.h5` embeddings from Stage 1 and a labels directory with one CSV per disease.

### CSV format

Each `{disease}.csv` must have columns:
- `case_id` — filename (e.g. `amos_0001.nii.gz`)
- `split` — `train`, `val`, or `test`
- `{disease}` — integer class label

### Run all 15 AMOS diseases

```bash
EMBEDS_DIR=/path/to/outputs \
LABELS_ROOT=/path/to/labels \
OUT_ROOT=/path/to/lp_results \
bash run_LP.sh
```

### Run a single disease

```bash
python run_LP.py \
  --embeds_dir /path/to/outputs \
  --labels_root /path/to/labels \
  --target splenomegaly \
  --out_dir /path/to/lp_results/splenomegaly/results
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--embeds_dir` | _(required)_ | Flat directory of `.h5` embedding files |
| `--labels_root` | _(required)_ | Directory containing `{disease}.csv` files |
| `--target` | `splenomegaly` | Disease name (must match CSV filename and label column) |
| `--out_dir` | _(required)_ | Output directory for results |
| `--epochs` | `1000` | Max training epochs |
| `--patience` | `50` | Early stopping patience |
| `--lrs` | 13 values 1e-5→0.1 | Learning rates to sweep |
| `--use_wandb` | off | Enable W&B logging |

### Outputs

Results are written to `--out_dir`:
- `val_report.csv` — per-head balanced accuracy, AUROC, F1, etc.
- `best_overall_balanced_acc*.pth` — top-K checkpoints
- `progress.png` — training curves

## Stage 3 — Per-scan Predictions

Generates a long-format CSV with per-scan predictions across all diseases:

```bash
python predict_LP.py \
  --embeds_dir /path/to/outputs \
  --lp_results_dir /path/to/lp_results \
  --labels_root /path/to/labels \
  --out_csv /path/to/predictions.csv
```

To run on a subset of diseases or when the label column name differs from the results directory name:

```bash
python predict_LP.py \
  --embeds_dir /path/to/luna/outputs \
  --lp_results_dir /path/to/lp_results \
  --labels_root /path/to/luna/labels \
  --diseases luna \
  --label_col lung_nodule_malignancy \
  --out_csv /path/to/luna_predictions.csv
```

Output columns: `filename`, `label`, `prediction`, `logit_class_0`, `logit_class_1`, `prob_class_0`, `prob_class_1`, `disease_name`

## Repository structure

```
.
├── extract_feat_LP.py   # CT-FM feature extraction (multi-GPU, HU windowing)
├── extract_feat_LP.sh   # Shell wrapper (Docker env vars)
├── run_LP.py            # Linear probing training & evaluation
├── run_LP.sh            # Runs LP for all 15 AMOS diseases
├── predict_LP.py        # Per-scan predictions from trained LP checkpoints
├── metrics/
│   ├── __init__.py
│   └── balanced_accuracy.py   # BalancedAccuracy torchmetrics class
├── Dockerfile           # Feature extraction container
└── requirements.txt
```

## Supported diseases (AMOS)

`splenomegaly`, `adrenal_hyperplasia`, `fatty_liver`, `cholecystitis`, `liver_calcifications`, `hydronephrosis`, `gallstone`, `liver_lesion`, `kidney_stone`, `liver_cyst`, `renal_cyst`, `atherosclerosis`, `colorectal_cancer`, `ascites`, `lymphadenopathy`
