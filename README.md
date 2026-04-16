# CT-FM Feature Extractor — CVPR 2026 CT FM Challenge

This repo contains the full pipeline for the [CVPR 2026 3D CT Foundation Model Challenge](https://github.com/kmin940/CVPR26-3DCTFMCompetition): feature extraction with the CT-FM model (SegResNet backbone) and linear probing evaluation across 15 diseases on the AMOS dataset.

## Overview

The pipeline has two stages:

1. **Feature extraction** — runs CT-FM on a directory of `.nii.gz` CT scans and writes 512-dim embeddings to `.h5` files
2. **Linear probing** — trains multi-head linear classifiers on those embeddings against disease labels, sweeping 13 learning rates

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

Run extraction:

```bash
docker run --gpus all \
  -v /path/to/inputs:/workspace/inputs \
  -v /path/to/outputs:/workspace/outputs \
  ct_fm_extractor bash extract_feat_LP.sh
```

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `INPUT_DIR` | `/workspace/inputs` | Directory of `.nii.gz` CT scans |
| `OUTPUT_DIR` | `/workspace/outputs` | Output directory for `.h5` embedding files |
| `MASKS_DIR` | _(unset)_ | Optional directory of segmentation masks |
| `NUM_CLASSES` | _(unset)_ | Number of classes (optional) |

### Without Docker

```bash
python extract_feat_LP.py -i /path/to/inputs -o /path/to/outputs
```

Each output `.h5` file contains a `y_hat` dataset of shape `(512,)`.

## Stage 2 — Linear Probing

Runs outside Docker. Expects `.h5` embeddings from Stage 1 and a labels directory with one CSV per disease.

### CSV format

Each `{disease}.csv` must have columns:
- `case_id` — filename (e.g. `amos_0001.nii.gz`)
- `split` — `train`, `val`, or `test`
- `{disease}` — integer class label

### Run all 15 diseases

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
| `--target` | `splenomegaly` | Disease name |
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

## Supported diseases

`splenomegaly`, `adrenal_hyperplasia`, `fatty_liver`, `cholecystitis`, `liver_calcifications`, `hydronephrosis`, `gallstone`, `liver_lesion`, `kidney_stone`, `liver_cyst`, `renal_cyst`, `atherosclerosis`, `colorectal_cancer`, `ascites`, `lymphadenopathy`

## Repository structure

```
.
├── extract_feat_LP.py   # CT-FM feature extraction script
├── extract_feat_LP.sh   # Shell wrapper (Docker env vars)
├── run_LP.py            # Linear probing training & evaluation
├── run_LP.sh            # Runs LP for all 15 diseases
├── metrics/
│   ├── __init__.py
│   └── balanced_accuracy.py   # BalancedAccuracy torchmetrics class
├── Dockerfile           # Feature extraction container
└── requirements.txt
```
