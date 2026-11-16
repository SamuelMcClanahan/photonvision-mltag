#!/usr/bin/env bash
set -euo pipefail

ONNX_MODEL="${1:-}"
OUT_DIR="${2:-qnn_apriltag_yolo}"
INPUT_NAME="${3:-images}"
INPUT_DIMS="${4:-1,3,640,640}"
BACKEND="${QNN_BACKEND:-/usr/lib/libQnnHtp.so}"

if [[ -z "$ONNX_MODEL" ]]; then
    echo "usage: $0 /path/to/model.onnx [out_dir] [input_name] [input_dims]" >&2
    exit 2
fi

if ! command -v qnn-onnx-converter >/dev/null 2>&1; then
    echo "qnn-onnx-converter is missing. Install the full Qualcomm QAIRT SDK converter tools first." >&2
    exit 1
fi

if ! command -v qnn-model-lib-generator >/dev/null 2>&1; then
    echo "qnn-model-lib-generator is missing. Install the full Qualcomm QAIRT SDK converter tools first." >&2
    exit 1
fi

mkdir -p "$OUT_DIR"

CPP_OUT="$OUT_DIR/apriltag_yolo_qnn.cpp"

qnn-onnx-converter \
    --input_network "$ONNX_MODEL" \
    --output_path "$CPP_OUT" \
    --input_dim "$INPUT_NAME" "$INPUT_DIMS"

qnn-model-lib-generator \
    -c "$CPP_OUT" \
    -b "$OUT_DIR/apriltag_yolo_qnn.bin" \
    -o "$OUT_DIR/model_lib"

MODEL_SO="$(find "$OUT_DIR/model_lib" -name "*.so" -type f | head -1)"
if [[ -n "$MODEL_SO" && -x "$(command -v qnn-context-binary-generator || true)" ]]; then
    qnn-context-binary-generator \
        --model "$MODEL_SO" \
        --backend "$BACKEND" \
        --binary_file apriltag_yolo_htp.bin \
        --output_dir "$OUT_DIR/context"
fi

echo "QNN conversion output: $OUT_DIR"
