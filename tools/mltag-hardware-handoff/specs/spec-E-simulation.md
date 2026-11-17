# Specification E: simulation of the uncompressed gray camera mode and a separate JPEG speed factor

File that you can edit:
- photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java
Read the class comment and the complete file first. The simulation runs the real capture library and the real AprilTag pipeline with a simulated 60 FPS 1600x1304 camera and a simulated NPU. It is a JUnit test that only runs with PV_MLTAG_SIM=1.

## Background
- The camera of the owner (ThriftyCam, USB 3.0) has two modes at 1600x1304 and 60 FPS: Motion-JPEG, and with firmware 5.10 or newer an uncompressed 8-bit gray mode (V4L2 GREY). In the gray mode the coprocessor does no JPEG decode. The capture provider opens such a camera with a pipeline of the form `v4l2src ! video/x-raw,format=GRAY8,width=1600,height=1304,framerate=60/1 ! appsink` and the native library gives the gray bytes of each frame.
- The simulated camera now is `multifilesrc location=frame_%04d.jpg ... caps=image/jpeg,framerate=60/1 ! identity name=pvcapturetime sleep-time=N ! appsink name=pvappsink ...` (see cameraPipeline). Each frame of the frame set is one JPEG file.
- The simulation has a "work" mode (PV_MLTAG_SIM_MODE=work): the camera and the NPU run in real time and each CPU stage does k times its work, where k is the CPU factor of the scenario (PV_MLTAG_SIM_SLOWDOWNS). The JPEG decode gets its extra work through GStreamerCaptureJNI.setTestExtraDecodePercent((k - 1) * 100).
- Measured with the same public benchmark (Phoronix Test Suite pts/tjbench, libjpeg-turbo, picture jpeg-test-1.JPG, 4:2:2 to BGR): the computer where the simulation was made gives 156 megapixels per second, and the RK3588 Cortex-A76 gives 97.4 (openbenchmarking.org result for a Mixtile Blade 3). So JPEG decode is 1.6 times slower on the RK3588, which is more than the general CPU factor of 1.1 to 1.3 from the 7-Zip benchmark.

## Items
### E1: camera format setting
- New setting PV_MLTAG_SIM_CAMERA_FORMAT with the values `mjpeg` (default, the present behavior) and `gray`.
- With `gray`: prepareScene writes each frame of the frame set as one raw file of exactly WIDTH x HEIGHT bytes (the 8-bit gray picture after applyLighting and drawFrameIndex, row after row, no header, no padding; 1600 is a multiple of 4 so the GStreamer row stride of GRAY8 is the width). File name `frame_%04d.gray`. No JPEG is written in this mode. The camera pipeline is `multifilesrc location="<dir>/frame_%04d.gray" index=<first> start-index=0 stop-index=<last> loop=true caps=video/x-raw,format=GRAY8,width=1600,height=1304,framerate=60/1 ! identity name=pvcapturetime sleep-time=<N> ! appsink name=pvappsink emit-signals=false max-buffers=1 drop=true sync=false`. Check with a short run that the capture library accepts it and that readFrameIndex reads the index from the delivered Mat.
- All other code paths stay: the reference run (prepareReference), the comparison of each result with the reference, calibrateCameraSleepMicros, both time modes. The SceneFrame field jpegBytes becomes the frame size in bytes of the active format; change its name to frameBytes and change the report text from "JPEG size" to "frame size (<format>)".
- The start line of the report ("mode: ...") must name the camera format.
- In `gray` mode GStreamerCaptureJNI.setTestExtraDecodePercent must be called with 0 (there is no JPEG decode).
- Disk space: 360 frames are about 750 MB. Write them to the same directory logic as now (tempDir or PV_MLTAG_SIM_FRAME_DIR).

### E2: separate JPEG factor
- New setting PV_MLTAG_SIM_JPEG_FACTOR (decimal). In work mode with camera format mjpeg, the extra JPEG decode work uses this factor: setTestExtraDecodePercent(round((jpegFactor - 1) * 100)), and not the general CPU factor of the scenario. If the setting is absent, the factor is the general CPU factor of the scenario (the present behavior). In the time-scaling mode the setting has no effect; say that in the comment.
- The report line of each scenario must show the JPEG factor that was used (for example `jpegFactor=1.6`) in mjpeg work mode.
- Add the measured numbers of the "Background" section above to the class comment, in short simple sentences.

### E3: check
- Compile: run `--tests "*MLTagEndToEndSimulation*"` without PV_MLTAG_SIM (the two tests are skipped, the class must compile).
- Then run two short simulations to prove that both formats work (these runs are permitted for this specification; each takes about 2 to 3 minutes):
  `PV_MLTAG_SIM=1 PV_MLTAG_SIM_MODE=work PV_MLTAG_SIM_CAMERAS=2 PV_MLTAG_SIM_SECONDS=10 PV_MLTAG_SIM_SLOWDOWNS=1.3 PV_MLTAG_SIM_SCENE_FRAMES=6 PV_OPENCV_THREADS=1 PV_GSTREAMER_CAPTURE_RT_PRIORITY=10 PV_GSTREAMER_CAPTURE_JNI_LIB=<library path from the Rules> ./gradlew :photon-core:cleanTest :photon-core:test --tests "*MLTagEndToEndSimulation.orangePi5" --console=plain -x spotlessApply -Porg.gradle.java.installations.paths=/usr/lib/jvm/java-25-openjdk-amd64`
  one time with `PV_MLTAG_SIM_CAMERA_FORMAT=gray` and one time with `PV_MLTAG_SIM_CAMERA_FORMAT=mjpeg PV_MLTAG_SIM_JPEG_FACTOR=1.6` in front. Both must end with BUILD SUCCESSFUL, and each scenario line must show `equalToReference=100.00%`. Put the lines that start with `[sim]` of both runs into your report (search the gradle output for "[sim]").
- Do not change how FPS, p99 or latency are computed.

## Rules for the implementer
- Work in <scratch>/wt in this directory. It is a separate git work tree (branch wt-impl) of the project. The specifications before yours are committed there. Edit only the files that this specification names. Do not commit, do not push, do not run spotless. The lead engineer does these.
- Do not run the end-to-end simulation (MLTagEndToEndSimulation) unless this specification says so.
- Build and test command for Java (JDK 25 is necessary, the option gives its path):
  `cd <scratch>/wt && ./gradlew :photon-core:test --tests "<TestClassPattern>" --console=plain -x spotlessApply -Porg.gradle.java.installations.paths=/usr/lib/jvm/java-25-openjdk-amd64`
  Only one gradle command can run at a time in this directory. If gradle says that the build directory is locked, wait and try again.
- The native capture library is built with: `cd <scratch>/wt-capbuild && make` (the build directory is configured for tools/jetson-gstreamer-capture-jni). Tests that use the library need the environment variable `PV_GSTREAMER_CAPTURE_JNI_LIB=<scratch>/wt-capbuild/libpv_gstreamer_capture_jni.so` in front of the gradle command. A second build without the JPEG decoder must also compile: `cd <scratch>/wt-capbuild-nojpeg && make`.
- The hard requirement of the owner: no change of tag IDs, tag corners, poses or capture timestamps, and no stale or mixed-up data. The gray image that the capture library gives must stay exactly the luma that libjpeg gives with the accurate integer DCT (or the exact bytes of a raw camera frame).
- Write comments in short simple English sentences that say WHY. No em dashes. Follow the style of the comments that are in the file.
- Each item needs a test that fails before the change and passes after it, as the item says. All tests that are in the named test classes now must pass after your work.
- If a method in this specification cannot work as written, stop on that item, keep the other items, and say exactly why in your report. Do not invent a different design.
- Final report (plain text): for each item: done or not done, the files and functions changed, the tests added, and the last lines of the test output.

