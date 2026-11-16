#!/usr/bin/env bash
set -uo pipefail

# Lists the hardware JPEG decoder elements that this board has. Give one of the
# names to PV_USB_GSTREAMER_HW_JPEG_DECODER. The capture library uses the luma
# plane of the decoder output directly, so no converter element is necessary for
# a monochrome camera. If the decoder gives no frame, PhotonVision uses the
# software decode of the capture library.

if ! command -v gst-inspect-1.0 >/dev/null 2>&1; then
    echo "gst-inspect-1.0 is not installed. Install the package gstreamer1.0-tools."
    exit 1
fi

found=0
# Rockchip MPP, V4L2 stateful decoders, Qualcomm, NVIDIA, VA-API.
for element in mppjpegdec v4l2jpegdec v4l2mjpegdec qtijpegdec qtivdec nvjpegdec nvv4l2decoder vajpegdec vaapijpegdec; do
    if gst-inspect-1.0 "$element" >/dev/null 2>&1; then
        found=1
        echo "FOUND   $element"
        gst-inspect-1.0 "$element" 2>/dev/null | grep -A12 "SRC template" | grep -E "format|memory" | head -4
    else
        echo "absent  $element"
    fi
done

echo
echo "Other elements that accept image/jpeg:"
gst-inspect-1.0 2>/dev/null | grep -i -E "jpeg.*dec|mjpeg" || true

if [[ "$found" -eq 0 ]]; then
    echo
    echo "No hardware JPEG decoder was found. The capture library decodes JPEG with libjpeg-turbo."
fi
