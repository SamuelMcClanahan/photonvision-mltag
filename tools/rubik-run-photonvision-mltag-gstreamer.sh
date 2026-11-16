#!/usr/bin/env bash
set -euo pipefail

# First-pass Rubik Pi 3 MLTag runner for the native GStreamer capture path.
#
# This mirrors the Jetson runner at the camera-capture layer, but inference still
# uses PhotonVision's existing Rubik/TFLite/QNN detector. A true QNN zero-copy
# path requires a new rubik_jni entrypoint that can consume a GstBuffer/DMABUF.

cd "${PV_RUBIK_HOME:-/home/photon/photonvision-pr2410}"

export PV_GSTREAMER_CAPTURE_JNI_LIB="${PV_GSTREAMER_CAPTURE_JNI_LIB:-/home/photon/pv-gstreamer-capture-jni/build/libpv_gstreamer_capture_jni.so}"
export PV_GSTREAMER_CAPTURE_WRAP_BUFFER="${PV_GSTREAMER_CAPTURE_WRAP_BUFFER:-true}"
export PV_GSTREAMER_CAPTURE_STRICT_ML_PTS="${PV_GSTREAMER_CAPTURE_STRICT_ML_PTS:-false}"
export PV_USB_GSTREAMER_BACKEND="${PV_USB_GSTREAMER_BACKEND:-qcom}"
export PV_USB_GSTREAMER_JNI_CAPTURE="${PV_USB_GSTREAMER_JNI_CAPTURE:-true}"
export PV_USB_GSTREAMER_PIXEL_FORMAT="${PV_USB_GSTREAMER_PIXEL_FORMAT:-MJPEG}"
export PV_USB_GSTREAMER_WIDTH="${PV_USB_GSTREAMER_WIDTH:-1280}"
export PV_USB_GSTREAMER_HEIGHT="${PV_USB_GSTREAMER_HEIGHT:-800}"
export PV_USB_GSTREAMER_FPS="${PV_USB_GSTREAMER_FPS:-120}"
export PV_USB_GSTREAMER_IO_MODES="${PV_USB_GSTREAMER_IO_MODES:-2,4,0}"
export PV_USB_GSTREAMER_OPEN_ATTEMPTS="${PV_USB_GSTREAMER_OPEN_ATTEMPTS:-3}"
export PV_USB_GSTREAMER_REOPEN_COOLDOWN_MS="${PV_USB_GSTREAMER_REOPEN_COOLDOWN_MS:-3000}"
export PV_USB_GSTREAMER_QCOM_JPEGPARSE="${PV_USB_GSTREAMER_QCOM_JPEGPARSE:-true}"

# The stock Rubik image observed during bring-up had Qualcomm runtime libraries
# installed but no gst-launch/gst-inspect CLI. Start with portable elements.
# Override these with Qualcomm-specific elements when the plugin package is present.
export PV_USB_GSTREAMER_QCOM_DECODER="${PV_USB_GSTREAMER_QCOM_DECODER:-jpegdec}"
export PV_USB_GSTREAMER_QCOM_CONVERTER="${PV_USB_GSTREAMER_QCOM_CONVERTER:-videoconvert}"
export PV_USB_GSTREAMER_CPU_FORMAT="${PV_USB_GSTREAMER_CPU_FORMAT:-BGR}"

# Optional experimental branch for future native QNN frame ingestion. This is
# not used by RubikObjectDetector yet and should stay off for profiling.
export PV_USB_GSTREAMER_QCOM_TEE="${PV_USB_GSTREAMER_QCOM_TEE:-false}"

if command -v gst-inspect-1.0 >/dev/null 2>&1; then
    gst-inspect-1.0 v4l2src >/dev/null 2>&1 || true
    gst-inspect-1.0 appsink >/dev/null 2>&1 || true
    gst-inspect-1.0 "${PV_USB_GSTREAMER_QCOM_DECODER}" >/dev/null 2>&1 || true
    gst-inspect-1.0 "${PV_USB_GSTREAMER_QCOM_CONVERTER}" >/dev/null 2>&1 || true
fi

if [[ -z "${PV_PHOTONVISION_JAR:-}" ]]; then
    dev_jar="$(find "$PWD/photon-server/build/libs" \
        -maxdepth 1 \
        -name 'photonvision-*-linuxarm64.jar' \
        -print 2>/dev/null | sort | tail -n 1 || true)"
    if [[ -n "$dev_jar" ]]; then
        PV_PHOTONVISION_JAR="$dev_jar"
    else
        PV_PHOTONVISION_JAR="/opt/photonvision/photonvision.jar"
    fi
fi

exec java -jar "$PV_PHOTONVISION_JAR"
