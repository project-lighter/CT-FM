#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
EMBEDS_DIR="${EMBEDS_DIR:-/workspace/outputs}"      # flat dir of .h5 files from extract_feat_LP.sh
LABELS_ROOT="${LABELS_ROOT:-/workspace/labels}"     # dir containing {disease}.csv files
OUT_ROOT="${OUT_ROOT:-/workspace/lp_results}"

disease_list=(
  splenomegaly adrenal_hyperplasia fatty_liver cholecystitis
  liver_calcifications hydronephrosis gallstone liver_lesion
  kidney_stone liver_cyst renal_cyst atherosclerosis
  colorectal_cancer ascites lymphadenopathy
)

for disease in "${disease_list[@]}"; do
    echo "Running linear probing for ${disease} ..."
    python run_LP.py \
        --embeds_dir "$EMBEDS_DIR" \
        --labels_root "$LABELS_ROOT" \
        --target "$disease" \
        --out_dir "$OUT_ROOT/${disease}/results"
done
