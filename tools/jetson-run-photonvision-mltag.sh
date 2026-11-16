#!/usr/bin/env bash
set -euo pipefail

cd "${PV_JETSON_HOME:-/home/samuel/pv-jetson}"

export PV_GSTREAMER_CAPTURE_JNI_LIB="${PV_GSTREAMER_CAPTURE_JNI_LIB:-/home/samuel/pv-jetson/gst-capture-jni-build/libpv_gstreamer_capture_jni.so}"
export PV_GSTREAMER_CAPTURE_STRICT_ML_PTS="${PV_GSTREAMER_CAPTURE_STRICT_ML_PTS:-true}"
export PV_GSTREAMER_CAPTURE_WRAP_BUFFER="${PV_GSTREAMER_CAPTURE_WRAP_BUFFER:-true}"
export PV_TENSORRT_CACHE_EGL_RESOURCE="${PV_TENSORRT_CACHE_EGL_RESOURCE:-true}"
export PV_TENSORRT_CUDA_GRAPH="${PV_TENSORRT_CUDA_GRAPH:-true}"
export PV_TENSORRT_CUDA_GRAPH_WARMUP="${PV_TENSORRT_CUDA_GRAPH_WARMUP:-60}"
export PV_TENSORRT_CUDA_GRAPH_PREPROCESS="${PV_TENSORRT_CUDA_GRAPH_PREPROCESS:-false}"
export PV_TENSORRT_GSTREAMER_INFERENCE_GRAPH="${PV_TENSORRT_GSTREAMER_INFERENCE_GRAPH:-true}"
export PV_TENSORRT_GSTREAMER_PREPROCESS_GRAPH="${PV_TENSORRT_GSTREAMER_PREPROCESS_GRAPH:-false}"
export PV_TENSORRT_GPU_LETTERBOX="${PV_TENSORRT_GPU_LETTERBOX:-true}"
export PV_TENSORRT_JNI_LIB="${PV_TENSORRT_JNI_LIB:-/home/samuel/pv-jetson/libpv_tensorrt_jni.so}"
export PV_USB_GSTREAMER_DECODER="${PV_USB_GSTREAMER_DECODER:-nvv4l2decoder}"
export PV_USB_GSTREAMER_OPEN_STAGGER_MS="${PV_USB_GSTREAMER_OPEN_STAGGER_MS:-1000}"
export PV_USB_GSTREAMER_NVMM_TEE="${PV_USB_GSTREAMER_NVMM_TEE:-true}"
if [[ "$PV_USB_GSTREAMER_NVMM_TEE" == "false" || "$PV_USB_GSTREAMER_NVMM_TEE" == "0" ]]; then
    export PV_TENSORRT_DISABLE_GSTREAMER_FRAME="${PV_TENSORRT_DISABLE_GSTREAMER_FRAME:-true}"
fi
# nvv4l2decoder can consume UVC MJPEG directly. Keeping jpegparse in front of it
# measured as a large extra timestamp-age buffer on the Orin Nano dual-camera path.
export PV_USB_GSTREAMER_JPEGPARSE="${PV_USB_GSTREAMER_JPEGPARSE:-false}"
export PV_USB_GSTREAMER_JNI_CAPTURE="${PV_USB_GSTREAMER_JNI_CAPTURE:-true}"

if [[ "${PV_USB_GSTREAMER_PREWARM:-true}" == "true" || "${PV_USB_GSTREAMER_PREWARM:-true}" == "1" ]]; then
    gst-inspect-1.0 v4l2src >/dev/null 2>&1 || true
    gst-inspect-1.0 nvv4l2decoder >/dev/null 2>&1 || true
    gst-inspect-1.0 nvvidconv >/dev/null 2>&1 || true
fi

if [[ -z "${PV_PHOTONVISION_JAR:-}" ]]; then
    dev_jar="$(find /home/samuel/pv-photonvision/photon-server/build/libs \
        -maxdepth 1 \
        -name 'photonvision-*-linuxarm64.jar' \
        -print 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "$dev_jar" ]]; then
        PV_PHOTONVISION_JAR="$dev_jar"
    else
        PV_PHOTONVISION_JAR="/home/samuel/pv-jetson/photonvision-jetson.jar"
    fi
fi

exec java -jar "$PV_PHOTONVISION_JAR"
