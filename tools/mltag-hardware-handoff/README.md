# ML AprilTag speed work: handoff for the hardware test

This file is for the person or agent that continues this work with the real
hardware. All the code is on the branch `claude/nice-shannon-q9zm9t`. Nothing in
this work was tested on real hardware. All numbers below are from a simulation
on an x86 computer.

## 1. Goal

- Camera: ThriftyCam (OG02B1B, monochrome, 1600x1304, 60 FPS, USB 3.0).
- One camera on an Orange Pi 5: 60 FPS.
- Two cameras on one Orange Pi 5: 55 FPS or more as an average, and the 99th
  percentile of the FPS value that PhotonVision shows must not be lower than
  40 FPS. The first target was 50 FPS or more.
- Rules: no frame decimation, no quantization of the model, no change that
  makes the tag data or the pose data worse.

## 2. What the code does now

| Area | Change | Main files |
|---|---|---|
| Inference lanes | Two or more frames are in inference at the same time on different NPU cores. The results stay in the order of the frames. | `AprilTagPipeline.java`, `AprilTagMLHybridPipe.java`, `InferenceOverlapController.java` |
| Capture library | Direct capture to an appsink. JPEG decode to a 1-channel image on a separate thread. Only the newest frame is kept, so no queue can grow. Capture time comes from the native clock. | `tools/jetson-gstreamer-capture-jni/src/GStreamerCaptureJNI.cpp`, `GStreamerFrame.java` |
| Capture provider | Reopen after a stall, no lock during the open, no frame leak. | `GStreamerUSBFrameProvider.java`, `CpuImageProcessor.java` |
| Model input | A 1-channel frame is resized one time and then copied to 3 channels. A self-check proves that the result is equal to the old operation. | `Letterbox.java`, `RknnObjectDetector.java`, `RubikObjectDetector.java` |
| Run scripts | Default settings for the board. | `tools/orangepi5-run-photonvision-mltag-gstreamer.sh`, `tools/rubik-run-photonvision-mltag-gstreamer.sh` |
| Simulation | Real capture library and real pipeline with a simulated camera and a simulated NPU. | `photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java` |

The folder `specs/` has the specification for each change (A to H).

## 3. What the simulation estimates

"NPU time" is the time for one inference while all three NPU cores are busy.

Two cameras on the Orange Pi 5, 2 lanes:

| NPU time | Average FPS, gray mode | Average FPS, MJPEG mode |
|---|---|---|
| 20.4 ms | 59 to 60 | 60 |
| 23.0 ms | 59 to 60 | 60 |
| 25.3 ms | 58 to 59 | 56 to 58 |
| 27.6 ms | 53 to 54 | 51 to 54 |
| 29.2 ms | 48 to 50 | 44 to 47 (with large JPEG frames) |

One camera: 59 to 60 FPS. With a slow NPU (29 to 32 ms) one camera needs
`PV_MLTAG_INFERENCE_LANES=3` for 60 FPS.

The tag IDs, corners and poses were equal to a calm reference run in 100% of
the results. The folder `sim-results/` has the report lines of the last runs.

## 4. Steps on the Orange Pi 5

Do the steps in this order. Write down each number.

### Step 1. Build

On the build computer:

```bash
./gradlew installArm64Toolchain
./gradlew :photon-server:shadowJar -PArchOverride=linuxarm64
```

Copy `photon-server/build/libs/photonvision-*-linuxarm64.jar` and the `tools`
folder to the board. On the board, build the capture library. The commands are
at the top of `tools/orangepi5-run-photonvision-mltag-gstreamer.sh`. The
configure step must print `JPEG decode in the capture library: ON`.

### Step 2. Find the camera video modes

```bash
v4l2-ctl -d /dev/video0 --list-formats-ext
```

- If the list has `GREY` at 1600x1304 and 60 FPS (camera firmware 5.10 or
  later), use `PV_USB_GSTREAMER_PIXEL_FORMAT=GRAY`. Then no JPEG decode is
  necessary. This is the mode with the most margin.
- If not, use the default (`MJPEG`).
- Connect each camera to a USB 3.0 port. Do not put two cameras on one hub
  before you know that the bandwidth is sufficient.

### Step 3. One camera, baseline

```bash
sudo PV_GSTREAMER_CAPTURE_TIMING_LOG_EVERY=600 PV_VISION_TIMING_LOG_EVERY=600 \
    tools/orangepi5-run-photonvision-mltag-gstreamer.sh
```

Use the ML AprilTag pipeline at 1600x1304 with a tag in the picture. Record:

1. Camera FPS, decoded FPS, delivered FPS, dropped frames and decode time
   (capture timing line).
2. The time of each pipeline stage, and the inference time (vision timing line).
3. The log line `ML AprilTag inference lanes: N frames in work`. It shows that
   the overlap is on.
4. The FPS and latency that PhotonVision shows.

### Step 4. Two cameras

Start the same script with two cameras connected. Record the same numbers for
each camera. The most important number is the inference time with two cameras,
because then all NPU cores are busy. Compare it with the table in section 3.

### Step 5. Compare with the threshold

Use the benchmark tool of the repository owner (average FPS and 99th percentile
of the FPS value that PhotonVision shows).

## 5. If the result is below the target

| Symptom | What to try |
|---|---|
| Camera FPS is below 60 | Exposure time is too long (dim light) or USB bandwidth. Set a manual exposure. Use a different USB port. Try the gray mode. |
| Decoded FPS is below camera FPS | JPEG decode is the limit. Use the gray mode, or `PV_USB_GSTREAMER_HW_JPEG_DECODER=mppjpegdec` (find it with `tools/gst-probe-hw-jpeg.sh`). The hardware decoder path is not tested. |
| Inference time with two cameras is above 25 ms | The NPU is the limit. Examine `PV_RKNN_CORE_MASK` (default -1). Examine the version of `librknnrt` (the JNI library ships 1.5.2). The remaining option is input without a copy (DMABUF), which is not done. |
| One camera is below 60 FPS and the inference is 29 ms or more | `PV_MLTAG_INFERENCE_LANES=3` |
| FPS is good but the time between results is not regular | Keep `PV_MLTAG_OVERLAP_RESERVE=0.3`. Make sure that the script runs as root, because the real-time priority of the capture threads needs it. |
| Latency must be lower and FPS has margin | `PV_MLTAG_OVERLAP_RESERVE=-0.25` (about 17 ms less, some results are late) |

To go back to the old behavior for a comparison:

- `PV_MLTAG_INFERENCE_LANES=1`: one frame at a time.
- `PV_USB_GSTREAMER_CAPTURE_MODE=legacy`: the old GStreamer pipeline.
- `PV_GSTREAMER_CAPTURE_JPEG_IDCT=fast`: gray pixels equal to the old pipeline.
- `PV_MLTAG_MONO_MODEL_INPUT=exact`: the old 3-channel model input operation.

## 6. Accuracy checks to do on hardware

1. Put a tag at a known distance and angle. Record the pose with the old
   settings (list above) and with the new settings. The poses must agree.
2. The default JPEG decode method (`accurate`) gives pixels that differ from the
   old pipeline by up to 6 gray levels. This is nearer to the camera picture,
   but it is a change. Use `fast` if the pixels must be equal to the old path.
3. Make sure that the capture time is correct: move a tag fast and compare the
   pose time with the robot odometry, or compare the latency value with a
   measured value.

## 7. Feed the measured numbers back into the simulation

The simulation can then give a better estimate for changes that are not on the
board yet.

```bash
PV_MLTAG_SIM=1 PV_MLTAG_SIM_MODE=work PV_MLTAG_SIM_CAMERAS=2 PV_MLTAG_SIM_DETAIL=1 \
PV_OPENCV_THREADS=1 PV_GSTREAMER_CAPTURE_RT_PRIORITY=10 \
PV_GSTREAMER_CAPTURE_JNI_LIB=/path/to/libpv_gstreamer_capture_jni.so \
PV_MLTAG_SIM_INFERENCE_MS=<measured ms> PV_MLTAG_SIM_CAMERA_FORMAT=gray \
./gradlew :photon-core:cleanTest :photon-core:test \
    --tests '*MLTagEndToEndSimulation.orangePi5' --console=plain -x spotlessApply
```

Other settings (all start with `PV_MLTAG_SIM_`): `SLOWDOWNS` (CPU factors),
`JPEG_FACTOR`, `MAX_JPEG_KB`, `LANES`, `SECONDS`, `PICTURE_FOLDERS` (real
pictures with YOLO label files), `NPU_CPU_MS`. The comment at the top of the
simulation file tells how to measure the CPU factors of the computer.

Inputs of the simulation that are not measured on hardware:

1. The inference time of the model with three busy NPU cores.
2. The CPU time that the NPU library uses for each inference (2 ms assumed).
3. The MJPEG frame size of the ThriftyCam.
4. The CPU speed ratio between the test computer and the RK3588 (from public
   benchmarks: 7-Zip and tjbench).

## 8. Rubik Pi 3

Use `tools/rubik-run-photonvision-mltag-gstreamer.sh` and the same steps. The
simulation estimate is 60 FPS for one camera and 56 to 59 FPS for two cameras.
Do not use `htp_performance_mode:2`.

## 9. Not done

- Model retrain with dark images (the training data is not in the repository).
- Correction of the int8 model in `rubik_jni`.
- DMABUF input to RKNN, and GStreamer frames directly to QNN.
- The hardware JPEG decoders (Rockchip and Qualcomm) are options in the code,
  but they are not tested.

## 10. Tests

```bash
PV_GSTREAMER_CAPTURE_JNI_LIB=/path/to/libpv_gstreamer_capture_jni.so \
./gradlew :photon-core:cleanTest :photon-core:test --console=plain -x spotlessApply
```

Last result on x86: 254 passed, 0 failed, 9 skipped (benchmark, simulation and
hardware tests). `BaseEqualityTest` compares the decode and pose results with a
golden file made from the code before this work (commit 412ccea).
