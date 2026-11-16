#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
DATA_YAML="${2:-}"
IMAGE_SIZE="${3:-640}"
OUT_DIR="${4:-qcom_model_export}"
VENV="${QCOM_AI_VENV:-$HOME/.venvs/qcom-ai}"

if [[ -z "$MODEL" || -z "$DATA_YAML" ]]; then
    echo "usage: $0 /path/to/model.pt /path/to/data.yaml [image_size] [out_dir]" >&2
    exit 2
fi

if [[ ! -x "$VENV/bin/yolo" ]]; then
    echo "missing YOLO export tooling at $VENV; expected $VENV/bin/yolo" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

"$VENV/bin/yolo" export \
    model="$MODEL" \
    format=tflite \
    imgsz="$IMAGE_SIZE" \
    int8=True \
    data="$DATA_YAML" \
    batch=1

EXPORTED="${MODEL%.*}_int8.tflite"
if [[ ! -f "$EXPORTED" ]]; then
    echo "INT8 TFLite export appeared to finish, but $EXPORTED was not found" >&2
    exit 1
fi

STABLE_TFLITE="$OUT_DIR/apriltag-yolo11n-640-int8.tflite"
cp "$EXPORTED" "$STABLE_TFLITE"

echo "INT8 TFLite export complete: $STABLE_TFLITE"
