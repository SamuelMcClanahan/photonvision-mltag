#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-}"
IMAGE_SIZE="${2:-640}"
OUT_DIR="${3:-qcom_model_export}"
VENV="${QCOM_AI_VENV:-$HOME/.venvs/qcom-ai}"

if [[ -z "$MODEL" ]]; then
    echo "usage: $0 /path/to/model.pt [image_size] [out_dir]" >&2
    exit 2
fi

if [[ ! -x "$VENV/bin/yolo" ]]; then
    echo "missing YOLO export tooling at $VENV; expected $VENV/bin/yolo" >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

BEFORE="$(mktemp)"
AFTER="$(mktemp)"
find "$(dirname "$MODEL")" -maxdepth 1 -name "*.onnx" -type f -print > "$BEFORE"

"$VENV/bin/yolo" export \
    model="$MODEL" \
    format=onnx \
    imgsz="$IMAGE_SIZE" \
    batch=1 \
    opset=17 \
    simplify=True \
    dynamic=False \
    project="$OUT_DIR" \
    name=onnx

find "$(dirname "$MODEL")" -maxdepth 1 -name "*.onnx" -type f -print > "$AFTER"
EXPORTED="$(comm -13 <(sort "$BEFORE") <(sort "$AFTER") | tail -1)"
if [[ -z "$EXPORTED" ]]; then
    EXPORTED="${MODEL%.*}.onnx"
fi

if [[ ! -f "$EXPORTED" ]]; then
    echo "ONNX export appeared to finish, but no exported .onnx was found" >&2
    exit 1
fi

STABLE_ONNX="$OUT_DIR/apriltag-yolo11n-640.onnx"
cp "$EXPORTED" "$STABLE_ONNX"

echo "ONNX export complete: $STABLE_ONNX"
