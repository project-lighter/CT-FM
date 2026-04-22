#!/bin/bash
set -e

# Default paths for Docker environment
INPUT_DIR="${INPUT_DIR:-/workspace/inputs}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/outputs}"
MASKS_DIR="${MASKS_DIR:-}"  # Optional masks directory
NUM_CLASSES="${NUM_CLASSES:-}"  # Default to 2 classes if not set
HU_MIN="${HU_MIN:-}"  # Optional HU window lower bound (default: -1024)
HU_MAX="${HU_MAX:-}"  # Optional HU window upper bound (default: 2048)
BATCH_SIZE=1

# Build command with optional masks_path
CMD="PYTHONPATH=. python extract_feat_LP.py -i \"$INPUT_DIR\" -o \"$OUTPUT_DIR\" --batch_size $BATCH_SIZE"

# Add masks_path argument if MASKS_DIR is set and not empty
if [ -n "$MASKS_DIR" ]; then
    CMD="$CMD --masks_path \"$MASKS_DIR\""
fi

# add num_classes argument if NUM_CLASSES is set and not empty
if [ -n "$NUM_CLASSES" ]; then
    CMD="$CMD --num_classes $NUM_CLASSES"
fi

if [ -n "$HU_MIN" ]; then
    CMD="$CMD --hu_min $HU_MIN"
fi

if [ -n "$HU_MAX" ]; then
    CMD="$CMD --hu_max $HU_MAX"
fi

# Run feature extraction
eval $CMD
