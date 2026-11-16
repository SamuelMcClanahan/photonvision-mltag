# Qualcomm QNN Tooling Notes

Installed on the Rubik Pi target:

- `qairt-tools`
- `qnn-tools`
- `libqnn-dev`
- `qairt-headers`
- `qcom-fastcv-binaries-dev`
- `libtensorflow-lite-qcom-dev`
- Qualcomm GStreamer ML plugins, including `qtimlvconverter`, `qtimltflite`, `qtimlqnn`, and `qtimlpostprocess`

Verified target commands:

- `qnn-context-binary-generator`
- `qnn-net-run`
- `qnn-throughput-net-run`
- `qnn-platform-validator`
- `qnn-profile-viewer`
- `snpe-net-run`
- `benchmark_model`

Not present in the Rubik Ubuntu packages:

- `qnn-onnx-converter`
- `qnn-model-lib-generator`
- `qairt-converter`

Those converter tools are part of the full Qualcomm QAIRT SDK download, not the target runtime packages. The direct SDK download endpoint currently returns 403 without an authenticated Qualcomm Software Center session.

Installed on the Mac in `~/.venvs/qcom-ai`:

- `qai-hub`
- `ultralytics`
- `onnx`
- `onnxsim`
- `onnxruntime`
- `torch`

Export a YOLO `.pt` to a fixed-shape ONNX:

```bash
tools/qcom-export-yolo-to-onnx.sh /path/to/model.pt 640 qcom_model_export
```

Native QNN route once the full QAIRT SDK converter is available:

```bash
qnn-onnx-converter \
  --input_network model.onnx \
  --output_path model_qnn.cpp \
  --input_dim images 1,3,640,640

qnn-model-lib-generator \
  -c model_qnn.cpp \
  -b model_qnn.bin \
  -o qnn_model_lib
```

Then the Rubik can run or package the generated model with:

```bash
qnn-context-binary-generator \
  --model qnn_model_lib/aarch64-ubuntu-gcc*/libmodel_qnn.so \
  --backend /usr/lib/libQnnHtp.so \
  --binary_file model_htp.bin \
  --output_dir qnn_context
```

For the current `.tflite` model, the nearer-term path is still `qtimltflite` with `libQnnTFLiteDelegate.so`. For a true `qtimlqnn` path, we need the generated QNN `.so` or context `.bin`.

Delegate option note from Rubik Pi 3 testing:

```bash
benchmark_model \
  --graph=/home/photon/qcom-models/apriltag-yolo11n-640-int8.tflite \
  --external_delegate_path=/usr/lib/libQnnTFLiteDelegate.so \
  --external_delegate_options=backend_type:htp
```

`backend_type:htp` successfully delegates the full graph. Adding `htp_performance_mode:2` caused delegate application failure with the installed QNN/TFLite delegate stack, so do not include that option in first-pass runtime scripts.

Current artifacts from `best(3).pt`:

- `qcom_model_export/apriltag-yolo11n-640.onnx`
- `qcom_model_export/apriltag-yolo11n-640-int8.tflite`
- Rubik copies:
  - `/home/photon/qcom-models/apriltag-yolo11n-640.onnx`
  - `/home/photon/qcom-models/apriltag-yolo11n-640-int8.tflite`
  - `/opt/photonvision/photonvision_config/models/apriltag-yolo11n-640-int8.tflite`

Rubik benchmark result with QNN TFLite delegate:

- `apriltag-yolo11n-640-int8.tflite` with `backend_type:htp`: about `8.9 ms` average inference in `benchmark_model`.
- Shipped `apriltagV4-yolo11.tflite` with `backend_type:htp`: about `9.4 ms` average inference in `benchmark_model`.
- Plain CPU/XNNPACK for the exported INT8 model was about `300 ms`, so QNN delegation is mandatory.

PhotonVision caveat:

- The new INT8 model loads and benchmarks through `benchmark_model`, and it is registered in PhotonVision as `AprilTag YOLO11n INT8`.
- Switching the live AprilTag pipeline to this model caused `rubik_jni` QNN graph execution/memory mapping errors. The live pipeline was reverted to `AprilTag V4`.
- This points at a `rubik_jni`/delegate-buffer compatibility issue, not a broken TFLite file. The QTI GStreamer path should use the proven minimal `backend_type:htp` delegate option and own its QNN graph/buffers directly.
