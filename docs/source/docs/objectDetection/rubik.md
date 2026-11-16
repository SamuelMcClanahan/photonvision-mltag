# Rubik Pi 3 Object Detection

## How it works

PhotonVision runs object detection on the Rubik Pi 3 by use of [TensorflowLite](https://github.com/tensorflow/tensorflow), and [this JNI code](https://github.com/PhotonVision/rubik_jni).

## Supported models

PhotonVision currently ONLY supports 640x640 Ultralytics YOLOv8 and YOLOv11 models trained and converted to `.tflite` format for QCS6490 SOCs! Other models require different post-processing code and will NOT work.

If you have a YOLO-Pro model from Edge Impulse, it can be converted using [this notebook](https://github.com/ramalamadingdong/yolo-pro-to-yolo11/tree/main).

## Converting Custom Models

:::{warning}
Only quantized models are supported, so take care when exporting to select the option for quantization.
:::

PhotonVision now ships with a {{ '[Python Notebook](https://github.com/PhotonVision/photonvision/blob/{}/scripts/rubik_conversion.ipynb)'.format(git_tag_ref) }} that you can use in [Google Colab](https://colab.research.google.com), [Kaggle](https://kaggle.com/code), or in a local environment. In Google Colab, you can simply paste the PhotonVision GitHub URL into the "GitHub" tab and select the `rubik_conversion.ipynb` notebook without needing to manually download anything.

Please ensure that the model you are attempting to convert is among the {ref}`supported models <docs/objectDetection/rubik:Supported Models>` and using the PyTorch format.

## Benchmarking

Before you can perform benchmarking, it's necessary to install `tensorflow-lite-qcom-apps` with apt.

By SSHing into your Rubik Pi and running this command, replacing `PATH/TO/MODEL` with the path to your model, `benchmark_model --graph=src/test/resources/yolov8nCoco.tflite --external_delegate_path=/usr/lib/libQnnTFLiteDelegate.so --external_delegate_options=backend_type:htp --external_delegate_options=htp_use_conv_hmx:1 --external_delegate_options=htp_performance_mode:2` you can determine how long it takes for inference to be performed with your model.

## Experimental GStreamer capture path

PhotonVision can run the Rubik Pi 3 camera through the native GStreamer USB capture provider by setting:

```bash
export PV_USB_GSTREAMER_JNI_CAPTURE=true
export PV_USB_GSTREAMER_BACKEND=qcom
export PV_GSTREAMER_CAPTURE_JNI_LIB=/path/to/libpv_gstreamer_capture_jni.so
```

The helper script `tools/rubik-run-photonvision-mltag-gstreamer.sh` sets the recommended first-pass defaults for the Rubik OV9281 USB camera path.

This path replaces CScore camera ownership with a low-buffer GStreamer `v4l2src -> appsink` capture path and carries GStreamer buffer timestamps into PhotonVision frame metadata. It still uses the existing Rubik `.tflite`/QNN object detector after the frame is exposed as an OpenCV `Mat`.

This is not yet a full QNN zero-copy inference path. The remaining native work is to add a `rubik_jni` entrypoint that can consume a GStreamer frame handle, `GstBuffer`, or DMABUF directly and run the same letterbox/inference/postprocess flow without returning through Java/OpenCV first. The Java detector has an explicit stub for this hook; set `PV_RUBIK_ENABLE_GSTREAMER_FRAME_STUB=true` to log when GStreamer-native frames are available but falling back to the current Mat path.

The QCOM backend starts with portable elements because Rubik images may ship Qualcomm runtime libraries without the GStreamer CLI/plugin packages installed. Use these overrides when Qualcomm decode/convert elements are available:

```bash
export PV_USB_GSTREAMER_QCOM_DECODE_STAGE='jpegparse ! <qualcomm-jpeg-decoder> ! <optional-caps>'
export PV_USB_GSTREAMER_QCOM_CPU_STAGE='<qualcomm-converter> ! video/x-raw,format=BGR,width=1280,height=800,framerate=120/1'
```

For profiling, leave `PV_USB_GSTREAMER_QCOM_TEE=false` until native QNN frame ingestion exists. The tee branch is only a future hook for carrying a second GStreamer buffer to inference and can add overhead today.
