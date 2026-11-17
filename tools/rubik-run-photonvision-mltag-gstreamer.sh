#!/usr/bin/env bash
set -euo pipefail

# Rubik Pi 3 (QCS6490) MLTag runner for the native GStreamer capture path.
#
# Capture: the camera buffer goes directly to the appsink. The capture library
# (tools/jetson-gstreamer-capture-jni) decodes the newest MJPEG frame to a
# 1-channel image on its own thread. No stage can hold more than one frame.
#
# Build the capture library on the board before the first run:
#   sudo apt-get install -y cmake g++ pkg-config libjpeg-turbo8-dev \
#       libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libopencv-dev
#   cmake -S tools/jetson-gstreamer-capture-jni -B /home/photon/pv-gstreamer-capture-jni/build
#   cmake --build /home/photon/pv-gstreamer-capture-jni/build -j4
# The configure step must print "JPEG decode in the capture library: ON".

pv_home="${PV_RUBIK_HOME:-/home/photon/photonvision-pr2410}"
if ! cd "$pv_home"; then
    echo "Cannot enter the PhotonVision directory: $pv_home. Set PV_RUBIK_HOME." >&2
    exit 1
fi

export PV_GSTREAMER_CAPTURE_JNI_LIB="${PV_GSTREAMER_CAPTURE_JNI_LIB:-/home/photon/pv-gstreamer-capture-jni/build/libpv_gstreamer_capture_jni.so}"

# Stop here with a clear message. Without the library PhotonVision starts with
# no camera and the log does not give the cause.
if [[ ! -r "$PV_GSTREAMER_CAPTURE_JNI_LIB" ]]; then
    echo "Capture library not found: $PV_GSTREAMER_CAPTURE_JNI_LIB. Build it (see the top of this script) or set PV_GSTREAMER_CAPTURE_JNI_LIB." >&2
    exit 1
fi
export PV_GSTREAMER_CAPTURE_WRAP_BUFFER="${PV_GSTREAMER_CAPTURE_WRAP_BUFFER:-true}"
export PV_GSTREAMER_CAPTURE_STRICT_ML_PTS="${PV_GSTREAMER_CAPTURE_STRICT_ML_PTS:-false}"
export PV_USB_GSTREAMER_BACKEND="${PV_USB_GSTREAMER_BACKEND:-qcom}"
export PV_USB_GSTREAMER_JNI_CAPTURE="${PV_USB_GSTREAMER_JNI_CAPTURE:-true}"

# Camera profile. Each camera on this coprocessor uses this video mode.
#   thriftycam: OG02B1B monochrome global shutter, 1600x1304 at 60 FPS
#   ov9281:     OV9281 monochrome global shutter, 1280x800 at 120 FPS
case "${PV_CAMERA_PROFILE:-thriftycam}" in
    ov9281)
        default_width=1280
        default_height=800
        default_fps=120
        ;;
    *)
        default_width=1600
        default_height=1304
        default_fps=60
        ;;
esac
# MJPEG works on USB 2 and USB 3. If "v4l2-ctl -d /dev/videoN --list-formats-ext"
# shows a raw format (YUYV or GREY) at the full frame rate on a USB 3 port, set
# PV_USB_GSTREAMER_PIXEL_FORMAT to YUYV or GRAY. Then no JPEG decode is necessary.
export PV_USB_GSTREAMER_PIXEL_FORMAT="${PV_USB_GSTREAMER_PIXEL_FORMAT:-MJPEG}"
export PV_USB_GSTREAMER_WIDTH="${PV_USB_GSTREAMER_WIDTH:-$default_width}"
export PV_USB_GSTREAMER_HEIGHT="${PV_USB_GSTREAMER_HEIGHT:-$default_height}"
export PV_USB_GSTREAMER_FPS="${PV_USB_GSTREAMER_FPS:-$default_fps}"
export PV_USB_GSTREAMER_IO_MODES="${PV_USB_GSTREAMER_IO_MODES:-2,4,0}"
export PV_USB_GSTREAMER_OPEN_ATTEMPTS="${PV_USB_GSTREAMER_OPEN_ATTEMPTS:-3}"
export PV_USB_GSTREAMER_OPEN_STAGGER_MS="${PV_USB_GSTREAMER_OPEN_STAGGER_MS:-1000}"
export PV_USB_GSTREAMER_REOPEN_COOLDOWN_MS="${PV_USB_GSTREAMER_REOPEN_COOLDOWN_MS:-3000}"

# The two cameras above are monochrome, so the 1-channel image has all the data
# of the camera. For a color camera set PV_USB_GSTREAMER_CPU_FORMAT=BGR.
export PV_USB_GSTREAMER_CPU_FORMAT="${PV_USB_GSTREAMER_CPU_FORMAT:-GRAY8}"

# auto:   direct capture first, then the legacy pipeline if direct gives no frame.
# direct: direct capture only.   legacy: jpegdec + videoconvert in GStreamer only.
export PV_USB_GSTREAMER_CAPTURE_MODE="${PV_USB_GSTREAMER_CAPTURE_MODE:-auto}"

# Optional hardware JPEG decoder element. It is used first if the board has it.
# Find candidates with: tools/gst-probe-hw-jpeg.sh
export PV_USB_GSTREAMER_HW_JPEG_DECODER="${PV_USB_GSTREAMER_HW_JPEG_DECODER:-}"

# Settings of the legacy fallback pipeline.
export PV_USB_GSTREAMER_QCOM_JPEGPARSE="${PV_USB_GSTREAMER_QCOM_JPEGPARSE:-true}"
export PV_USB_GSTREAMER_QCOM_DECODER="${PV_USB_GSTREAMER_QCOM_DECODER:-jpegdec}"
export PV_USB_GSTREAMER_QCOM_CONVERTER="${PV_USB_GSTREAMER_QCOM_CONVERTER:-videoconvert}"

# Optional experimental branch for future native QNN frame ingestion. This is
# not used by RubikObjectDetector yet and should stay off for profiling.
export PV_USB_GSTREAMER_QCOM_TEE="${PV_USB_GSTREAMER_QCOM_TEE:-false}"

# OpenCV divides each small image operation between all CPUs. In this pipeline
# that only adds thread wake-ups: with two cameras the simulation gave about
# 47 FPS without this setting and about 56 FPS with it. The results are equal.
export PV_OPENCV_THREADS="${PV_OPENCV_THREADS:-1}"

# Real-time priority (SCHED_FIFO) for the capture threads: the thread that takes
# the frames from the camera and the JPEG decode thread. They get a CPU
# immediately when a frame arrives, also when the vision pipelines use all
# CPUs. In the two-camera simulation this increased the frame rate that 99% of
# the frames keep from about 41 FPS to about 46 FPS. It needs root or
# CAP_SYS_NICE. Set it to 0 to use the normal thread policy.
export PV_GSTREAMER_CAPTURE_RT_PRIORITY="${PV_GSTREAMER_CAPTURE_RT_PRIORITY:-10}"

# Frame-flow diagnosis. Set this to 600 to print one line for each 600 frames:
# camera FPS, decoded FPS, delivered FPS, dropped frames, frame age, decode time.
export PV_GSTREAMER_CAPTURE_TIMING_LOG_EVERY="${PV_GSTREAMER_CAPTURE_TIMING_LOG_EVERY:-0}"
export PV_VISION_TIMING_LOG_EVERY="${PV_VISION_TIMING_LOG_EVERY:-0}"

if command -v gst-inspect-1.0 >/dev/null 2>&1; then
    gst-inspect-1.0 v4l2src >/dev/null 2>&1 || true
    gst-inspect-1.0 appsink >/dev/null 2>&1 || true
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

# CPUs 4-7 are the Cortex-A78 cores of the QCS6490. The benchmark of 2026-07-09
# used this setting. Set PV_CPU_AFFINITY to an empty value to use all CPUs.
cpu_affinity="${PV_CPU_AFFINITY-4-7}"
if [[ -n "$cpu_affinity" ]] && command -v taskset >/dev/null 2>&1; then
    # The kernel refuses a CPU list with CPUs that are not online or not in the
    # cpuset of a container. Then start without the limit, not with no PhotonVision.
    if taskset -c "$cpu_affinity" true >/dev/null 2>&1; then
        exec taskset -c "$cpu_affinity" java -jar "$PV_PHOTONVISION_JAR"
    fi
    echo "Warning: taskset does not accept the CPU list '$cpu_affinity'. Starting java on all CPUs." >&2
fi
exec java -jar "$PV_PHOTONVISION_JAR"
