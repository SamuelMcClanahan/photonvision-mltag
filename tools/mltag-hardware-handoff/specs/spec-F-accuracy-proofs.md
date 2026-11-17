# Specification F: proofs that the picture data and the model input are correct

Files that you can edit:
- photon-core/src/main/java/org/photonvision/vision/objects/Letterbox.java, RknnObjectDetector.java, RubikObjectDetector.java
- tools/jetson-gstreamer-capture-jni/src/GStreamerCaptureJNI.cpp
- photon-core/src/main/java/org/photonvision/gstreamer/GStreamerCaptureJNI.java and GStreamerCapture.java
- photon-core/src/test/java/org/photonvision/vision/objects/LetterboxMonochromeTest.java
- photon-core/src/test/java/org/photonvision/gstreamer/GStreamerCaptureTest.java
- photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java and photon-core/src/test/java/org/photonvision/benchmark/MLTagStageBenchmark.java (only the model input preparation, item F1 part C)
- tools/orangepi5-run-photonvision-mltag-gstreamer.sh and tools/rubik-run-photonvision-mltag-gstreamer.sh (comments and one setting, item F2)
- new files for item F3 (see there)

## Decisions of the lead engineer
- F1 (model input): implement the corrected method of the check (parts A, B, C and D). The fast 1-channel path is used only when the self-check proved byte equality on the OpenCV build that runs. In part C do not run the simulation; only change its code and compile it.
- F2 (JPEG decode method): the owner gets BOTH methods, and the DEFAULT STAYS THE ACCURATE METHOD (JDCT_ISLOW). This is different from the corrected method of the check, which makes the fast method the default. Reason: the accurate decode is nearer to the camera picture, so it does not make the tag data worse. Implement:
  - Setting PV_GSTREAMER_CAPTURE_JPEG_IDCT, read one time when a capture is created. Values `accurate` (default, also for no value and for unknown values) = JDCT_ISLOW. Value `fast` = JDCT_IFAST, which is the default method of the GStreamer jpegdec element, so the gray pixels are then equal to the old capture pipeline.
  - Native: Capture gets the field, decodeJpeg gets the method as a parameter, both decodeJpeg calls of frameFromJpegSample pass it (first frame in grab and the worker). A JNI function setJpegIdctAccurate(long capturePtr, boolean accurate) and capability bit 32 (CAP_JPEG_IDCT) as in the corrected method, so a test can select the method for one capture. The test-only extra decode must use the same method.
  - Tests in GStreamerCaptureTest on a monochrome 4:2:2 JPEG of 1600x1304 at quality 90 (make it as the existing tests make their pictures): (a) fast mode, OUTPUT_GRAY: pixel-exact against the old pipeline `jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=GRAY8`. (b) fast mode: pixel-exact against the exact old path `jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=BGR` followed by Imgproc.cvtColor BGR2GRAY. (c) accurate mode (default): the present assertions stay (pixel-exact against Imgcodecs.imread, and directGrayImageIsNotLessAccurateThanLegacyImage). If (a) or (b) is not pixel-exact, do not weaken the test: report the largest difference and the number of different pixels, and keep the test with an assertion on what you measured plus a comment.
  - Print in the test log (System.out, prefix "[capture]") the mean squared difference and the largest difference between the accurate and the fast picture.
  - Run scripts: add the setting with its default `accurate` and a comment of 3 to 5 short lines that says what each value does.
- F3 (equality with the base commit for tag decode and pose): see the item text below.
- Run at the end: `--tests "*LetterboxMonochromeTest" --tests "*GStreamerCaptureTest" --tests "*AprilTagROIParallelDecodeTest" --tests "*AprilTagROIDecodePipeTest" --tests "*AprilTagParallelPoseTest" --tests "*BaseEqualityTest" --tests "*MLTagEndToEndSimulation*" --tests "*MLTagStageBenchmark*"` with PV_GSTREAMER_CAPTURE_JNI_LIB set (the simulation and benchmark tests are skipped without their switches; they must compile). All must pass.

## Rules for the implementer
- Work in <scratch>/wt in this directory. It is a separate git work tree (branch wt-impl) of the project. The specifications before yours are committed there. Edit only the files that this specification names. Do not commit, do not push, do not run spotless. The workflow of the lead engineer commits after the check.
- Do not move or rename files. Do not work in any other copy of the project, except where this specification tells you to make a second work tree of the base commit.
- Build and test command for Java (JDK 25 is necessary, the option gives its path):
  `cd <scratch>/wt && ./gradlew :photon-core:test --tests "<TestClassPattern>" --console=plain -x spotlessApply -Porg.gradle.java.installations.paths=/usr/lib/jvm/java-25-openjdk-amd64`
  Only one gradle command can run at a time in one directory. If gradle says that the build directory is locked, wait and try again.
- The native capture library is built with: `cd <scratch>/wt-capbuild && make` (the build directory is configured for tools/jetson-gstreamer-capture-jni of the work tree). Tests that use the library need the environment variable `PV_GSTREAMER_CAPTURE_JNI_LIB=<scratch>/wt-capbuild/libpv_gstreamer_capture_jni.so` in front of the gradle command. A second build without the JPEG decoder must also compile: `cd <scratch>/wt-capbuild-nojpeg && make`.
- The hard requirement of the owner: no interference with accuracy. Tag IDs, tag corners, poses and capture timestamps must not get worse, and no stale or mixed-up data.
- Write comments in short simple English sentences that say WHY. No em dashes. Follow the style of the comments that are in the file.
- Each item needs a test that fails before the change and passes after it, as the item says. All tests that are in the named test classes now must pass after your work.
- If a method in this specification cannot work as written, stop on that item, keep the other items, and say exactly why in your report. Do not invent a different design.
- Final report (plain text): for each item: done or not done, the files and functions changed, the tests added, and the last lines of the test output.

## Items
### F1: Model input for a 1-channel frame is proven equal to the base BGR input only on the x86-64 host, not on the aarch64 targets
- Location: /home/user/photonvision-mltag/photon-core/src/main/java/org/photonvision/vision/objects/RknnObjectDetector.java:138
- Failure scenario: Orange Pi 5 or Rubik Pi 3 (aarch64) with the run-script default PV_USB_GSTREAMER_CPU_FORMAT=GRAY8. The AprilTag pipeline gets a CV_8UC1 frame. Letterbox.letterbox resizes 1 channel and the detector then does GRAY2BGR. The base code resized a CV_8UC3 image. On ARM builds of OpenCV 4.10 with the Carotene HAL (ON by default for AArch64 unless the build sets WITH_CAROTENE=OFF), cv::resize INTER_LINEAR for CV_8U goes to carotene resizeLinearOpenCV only for 1 and 4 channels. A 3-channel image uses the generic code, and for an exact 2:1 scale (OV9281 profile, 1280x800 to 640x400) the generic code changes to the 2x2 area average. Carotene calculates the coefficients in float32 and has an approximate NEON path for scales of 2 or less. If that HAL is active, some tensor bytes differ by 1 from the base tensor. The quantized YOLO output can then change, an ROI rectangle can move by 1 pixel or a box near the confidence threshold can appear or disappear, and the tag corners and poses are not equal to base. STATUS: not proven. I could not inspect the aarch64 OpenCV build or run on a target. What is proven is that the equality proof was made only on x86-64.
- Evidence: RknnObjectDetector.detect lines 138-152 and RubikObjectDetector.detect lines 136-150: letterbox on the input Mat, then cvtColor GRAY2BGR if letterboxed.channels()==1. Letterbox.java line 85: Imgproc.resize(frame, resizedScratch, newUnpad, Imgproc.INTER_LINEAR) (the 4th argument is fx, so the interpolation is the default INTER_LINEAR, same as base). LetterboxMonochromeTest is the only proof of the comment 'This gives the same bytes as a resize of a 3-channel image', and photon-core/build/NativeMain contains only linux/x86-64 OpenCV libraries. tools/orangepi5-run-photonvision-mltag-gstreamer.sh line 53 and tools/rubik-run-photonvision-mltag-gstreamer.sh line 54 make GRAY8 the default, so this is the production path. rubik_jni.cpp line 379-383 rejects a 1-channel Mat, so base could only run the model on BGR.
- First method from the review: Step 1 (decides if a code change is necessary): run LetterboxMonochromeTest on the Orange Pi 5 and on the Rubik Pi 3 with the linuxarm64 natives. If monochromeModelInputEqualsColorModelInput passes for 1600x1304 and 1280x800, write that in the test comment and stop. Step 2 (if it fails, or if a target run is not possible): in Letterbox.java add 'public static boolean monochromeResizeIsExact(Size frameSize, Size newShape, Scalar color)' with a static ConcurrentHashMap cache keyed by the four sizes. On a cache miss: make a CV_8UC1 Mat of frameSize from java.util.Random with a fixed seed, and blur the top half (same as randomGray in the test). Path A (base operation): cvtColor GRAY2BGR, then letterbox the 3-channel image. Path B (fast): letterbox the 1-channel image, then cvtColor GRAY2BGR. exact = Core.norm(A, B, Core.NORM_INF) == 0. Release the Mats, store the result, log one info line. In RknnObjectDetector.detect and RubikObjectDetector.detect, before the letterbox call: if in.channels()==1 and monochromeResizeIsExact is false, do Imgproc.cvtColor(in, grayAsBgr, COLOR_GRAY2BGR) into a new retained scratch Mat (release it in release()) and letterbox grayAsBgr. That is exactly the base operation on an image with three equal channels. Keep the existing fast branch for the true case. The simplest alternative is to always convert first. Test: extend LetterboxMonochromeTest with a package-private overload that forces the answer true and false, and assert for all five sizes that the model input bytes are equal to the base path in both cases.
- Result of the independent check: real=True, confidence=medium. Reasoning: I could not refute it. The repository part of the scenario is confirmed by reading the code; the OpenCV part cannot be read on this machine and rests on my memory of the OpenCV 4.10 source. I estimate about 70% that the bytes really differ on the boards.

Confirmed in the working tree:
- Both run scripts set PV_USB_GSTREAMER_CPU_FORMAT to GRAY8 by default (tools/orangepi5-run-photonvision-mltag-gstreamer.sh, tools/rubik-run-photonvision-mltag-gstreamer.sh). At base the Rubik script and the Java default were BGR.
- VisionRunner calls frameSupplier.requestMonochromeInput(pipeline instanceof AprilTagPipeline). GStreamerUSBFrameProvider.grabGStreamerFrame (lines 92-95) then keeps the CV_8UC1 image.
- AprilTagMLHybridPipe (lines 174 and 187) gives frame.colorImage to AprilTagROIDetectionPipe, which calls detector.detect with no conversion.
- RknnObjectDetector.detect (138-152) and RubikObjectDetector.detect (136-150) call Letterbox.letterbox on the 1-channel Mat and do GRAY2BGR after it. Letterbox line 85 is Imgproc.resize with the default INTER_LINEAR.
- Base resized a 3-channel Mat. rubik_jni.cpp 379-383 rejects other channel counts, so base could only run on BGR.
- The only proof of "same bytes" is LetterboxMonochromeTest. It runs with the host natives only: the Gradle cache has only opencv-cpp-4.10.0-3-linuxx86-64, and photon-core/build has no arm64 files.
- No code has a platform guard or a self-check (no match for carotene, INTER_LINEAR_EXACT, or a similar check).

Not confirmed here (no OpenCV source and no aarch64 binary on this machine):
- From memory of OpenCV 4.10: cv::hal::resize calls CALL_HAL(resize, cv_hal_resize, ...) before the generic code. The Carotene HAL (TEGRA_RESIZE) takes CV_8U INTER_LINEAR only when isResizeLinearOpenCVSupported is true, which is for 1 and 4 channels. A 3-channel image goes to the generic code.
- Carotene gets the scale factors as float32. For scales of 2 or less with 1 channel it uses downsample_bilinear_8uc1, which is compiled with USE_CORRECT_VERSION 0. The generic code calculates the coefficients in double.
- For 1600x1304 to 640x522 the vertical scale 1304/522 is not exact in float32, so some row coefficients can differ by 1/2048 and some output bytes by 1.
- WITH_CAROTENE is ON by default for AArch64. I do not remember that the WPILib thirdparty-opencv build sets it OFF, but I could not inspect that build.

An existing test does not cover the fault: it proves the equality only on x86-64, where both channel counts use the same generic code.

The size of the effect is small: a difference of 1 in a small number of tensor bytes, which can move a quantized box or a box at the confidence threshold on a small part of the frames. But the code comment and the test claim byte equality, and the owner requires no change, so the gap is a real fault against that requirement.

The proposed fix is correct in principle but not complete:
1. "Run the test on the board one time and stop" protects only one OpenCV binary and five sizes, and nobody can do it now. The runtime self-check must be unconditional.
2. The probe picture is blurred in the top half. Smooth content hides a coefficient difference of 1/2048, so the probe can miss a mismatch in those rows.
3. MLTagEndToEndSimulation (lines 1186-1201) and MLTagStageBenchmark (lines 303-316) have their own copy of the input preparation. If the board uses the slower exact path, the FPS estimate is still made with the fast path, which is a wrong measurement.

Side note, not part of this verdict: the new capture library decodes with JDCT_ISLOW (GStreamerCaptureJNI.cpp line 1286). The base pipeline used GStreamer jpegdec, whose default is ifast as far as I remember. Pixel equality with base at capture is a separate question for the capture area.
- Corrected method from the check (use THIS method unless the "Decisions" section says something different): Make one shared helper for the model input, with a self-check that always runs on the real platform. Use it in both detectors and in the simulation.

A. photon-core/src/main/java/org/photonvision/vision/objects/Letterbox.java

1. Add `public static final class ModelInputScratch` with four retained Mats (grayAsBgr, resized, letterboxed, letterboxedBgr) and `release()`.

2. Add `public record ModelInput(Letterbox scale, Mat image)`.

3. Add `public static ModelInput prepareModelInput(Mat in, ModelInputScratch s, Size inputSize, Scalar color)`:
   - `in.channels() != 1`: `letterbox(in, s.resized, s.letterboxed, inputSize, color)`; return (scale, s.letterboxed). This is the same as now.
   - `in.channels() == 1` and `monochromeResizeIsExact(in.cols(), in.rows(), inputSize, color)` is true: the fast path of today. Letterbox the 1-channel Mat into s.letterboxed, `cvtColor(s.letterboxed, s.letterboxedBgr, COLOR_GRAY2BGR)`, return (scale, s.letterboxedBgr).
   - Otherwise the base operation: `cvtColor(in, s.grayAsBgr, COLOR_GRAY2BGR)`, then `letterbox(s.grayAsBgr, s.resized, s.letterboxed, inputSize, color)`, return (scale, s.letterboxed).

4. Add `static boolean monochromeResizeIsExact(int width, int height, Size inputSize, Scalar color)`:
   - Read `EnvUtils.string("PV_MLTAG_MONO_MODEL_INPUT", "auto")`. The value "exact" returns false always (always the base operation). Do not add a production value that forces the fast path.
   - If width or height is 0 or less, return true.
   - Otherwise use a static `ConcurrentHashMap` with an immutable key (a record of the four integer sizes, not the mutable `Size`) and `computeIfAbsent(key, probe)`.

5. Add package-private `static boolean probeMonochromeResize(int width, int height, Size inputSize, Scalar color, MonoFastPath fastPath)`. `MonoFastPath` is a package-private functional interface `(Mat gray, Mat outBgr)`; production passes the real fast path.
   - Probe picture P1: CV_8UC1, the full frame filled by `java.util.Random` with a fixed seed.
   - Probe picture P2: P1 after `GaussianBlur` 9x9 on the full frame.
   - For each picture: A = `cvtColor` GRAY2BGR, then the 4-argument `letterbox`. B = `fastPath`.
   - The result is exact only if, for both pictures, size and type are equal and `Core.norm(A, B, Core.NORM_INF) == 0`.
   - Release all Mats.
   - Write one log line for each size: info if exact; warn if not exact, with the sizes, the number of different bytes, and the text that the 3-channel base operation is used and costs more time.

B. RknnObjectDetector.detect and RubikObjectDetector.detect
- Replace the three scratch Mats with one `ModelInputScratch`.
- Call `Letterbox.prepareModelInput(in, scratch, this.inputSize, ColorHelper.colorToScalar(Color.GRAY))`.
- Keep the size check on `input.image()`. Give `input.image().getNativeObjAddr()` to the JNI. Use `input.scale().resizeDetections(...)`.
- Release the scratch in `release()`.
- Replace the comment "This gives the same bytes..." with: the fast path is used only when the self-check proved it on this OpenCV build.

C. Simulation and benchmark
- MLTagEndToEndSimulation.java (about lines 1186-1201) and MLTagStageBenchmark.java (about lines 303-316): call `Letterbox.prepareModelInput` in place of the copied steps.
- Print in the report which path was active.
- Add one run with `PV_MLTAG_MONO_MODEL_INPUT=exact`, so the FPS estimate also exists for a board where the self-check fails.
- Run these only when the build lock is free.

D. Tests in LetterboxMonochromeTest
Add a package-private overload `prepareModelInput(in, s, size, color, boolean useFastPath)` for these tests.
1. `probeDetectsInexactFastPath`: give `probeMonochromeResize` a fast path that calls the real one and then adds 1 to one byte of the output. Expect false. With the real fast path, expect true (comment: proven on this host only).
2. `basePathEqualsBase`: for 1600x1304, 1280x800, 640x480, 640x640 and 1304x1600, call the overload with `useFastPath=false` two times with the same scratch. Assert the bytes equal `Letterbox.letterbox(GRAY2BGR(gray), expected, ...)` and that dx, dy and scale are equal. Then give a 3-channel Mat with different channels to the same scratch and assert it equals the 4-argument `letterbox`.
3. `fastPathEqualsBaseOnThisHost`: the assertion of today, through the helper with `useFastPath=true`.
4. `probeResultIsCached`: call `monochromeResizeIsExact` two times for one size and assert with a counter that the probe ran one time.

E. Check on a board (does not block the code change)
- Start PhotonVision on the Orange Pi 5 and the Rubik Pi 3 and read the one log line for each size.
- `Core.getBuildInformation()` (already printed by rubik_jni RubikTest) shows on the line "Custom HAL" whether Carotene is in the build.

Notes
- The slower path adds a full-resolution GRAY2BGR (about 6.3 MB written for each 1600x1304 frame) and a 3-channel resize for each frame. Its time is not measured here.
- The self-check runs one time for each frame size on the first detect call. That first inference time is an outlier, the same as the model warm-up.
- The simplest valid alternative is to always convert to BGR before the letterbox. Then the tests of D.2 are sufficient, but the model-input time gain of the gray path is lost on all platforms.

### F2: Outside my area (capture): the direct JPEG decode gives pixel values that differ from the base jpegdec pipeline, so corners cannot be equal to base
- Location: /home/user/photonvision-mltag/tools/jetson-gstreamer-capture-jni/src/GStreamerCaptureJNI.cpp:1286
- Failure scenario: Same MJPEG frame from a monochrome camera. Base: 'jpegparse ! jpegdec ! videoconvert ! BGR', then BGR2GRAY for each region. New default (auto mode, direct candidate first): libjpeg-turbo in the capture library with dct_method = JDCT_ISLOW. GStreamer jpegdec uses idct-method=ifast by default (from my knowledge of GStreamer, not from this repository). The gray values that the tag detector and the model get differ by some gray levels in many pixels. The corners and poses then differ from base below 1 pixel, and a tag near the decision-margin threshold can change. This is a change of tag data compared with base, even if the new picture is nearer to the camera picture.
- Evidence: GStreamerCaptureJNI.cpp line 1285-1286: 'The accurate integer inverse DCT. It gives the reference pixel values.' cinfo->dct_method = JDCT_ISLOW. GStreamerCaptureTest.directGrayImageIsNotLessAccurateThanLegacyImage lines 288-304 compares the direct image with the legacy pipeline image and only asserts 'largestDifference <= 6' with the comment 'the difference is only decoder rounding'. So the repository's own test accepts that the direct picture is not equal to the base picture. GStreamerUSBFrameProvider.buildCandidates lines 324-336 puts the direct candidate before the legacy pipeline.
- First method from the review: In decodeJpeg set cinfo->dct_method = JDCT_IFAST as the default, and read an environment variable (for example PV_GSTREAMER_CAPTURE_JPEG_IDCT=islow) one time at open to select the accurate method. Then run directGrayImageIsNotLessAccurateThanLegacyImage and print largestDifference. Expected: 0, because a monochrome JPEG has neutral chroma and the legacy path then gives R=G=B=Y. If it is 0, change the assertion in that test to 'largestDifference == 0' for the default mode and keep the accuracy assertion only for the islow mode. If it is not 0 with IFAST, the remaining difference comes from videoconvert; report the number to the owner as a known difference from base. The reviewer of the capture area must confirm this item.
- Result of the independent check: real=True, confidence=high. Reasoning: I could not refute this. The fault is real as a deviation from the owner's rule "no change of tag corners and poses compared with base". The new picture is not wrong (it is nearer to the camera picture), so the practical effect is small, but it is not equal to base.

Trace:
1. Base (412ccea), /home/user/photonvision-mltag/photon-core/src/main/java/org/photonvision/vision/frame/provider/GStreamerUSBFrameProvider.java: the QCOM and generic backends have one pipeline shape, "v4l2src ! image/jpeg ! queue ! jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! appsink". The base Rubik run script sets QCOM_DECODER=jpegdec and CPU_FORMAT=BGR. AprilTagROIDecodePipe then does BGR2GRAY for each region.
2. Now: buildCandidates() (lines 324-346) puts the "direct" candidate before "legacy" when the capture mode is not "legacy". The Java default is "auto", and the two run scripts set "auto" and CPU_FORMAT=GRAY8. So the default path is "v4l2src ! image/jpeg ! appsink" with decode in the capture library.
3. /home/user/photonvision-mltag/tools/jetson-gstreamer-capture-jni/src/GStreamerCaptureJNI.cpp, decodeJpeg() line 1286: cinfo->dct_method = JDCT_ISLOW, for gray and for BGR output. The two callers (first frame in grab() at line 1873, worker at line 1514) go through frameFromJpegSample(), so each direct frame uses ISLOW.
4. Measured proof that the pictures differ: the test directGrayImageIsNotLessAccurateThanLegacyImage (GStreamerCaptureTest.java lines 247-305) compares the direct picture with the base pipeline picture. Logs of earlier runs on this machine (for example <scratch>/gradle-test12.log line 86) show: "direct=7.0457 legacy=7.3069, largest direct-to-legacy pixel difference=6.0". The assertion limit "<= 6" is exactly the measured value. The different mean squared errors show that many pixels differ, not only one.
5. No test shows equal tag data between the direct path and the base path. The five "pixel-exact" tests compare with OpenCV imread (libjpeg default ISLOW), not with the base pipeline. MLTagEndToEndSimulation compares a load run with a calm run of the same decoder, so it cannot find this.
6. Result: for the same JPEG bytes the detector gets gray values that differ by up to 6 levels from base. The AprilTag quad fit and edge refinement use pixel gradients, so the corner coordinates and the poses are not bit-equal to base, and a tag near the decision margin limit can change. Commit 34aaf14 has the title "no change to the tag data", which is not true for the capture step.

Not confirmed from the repository: that the cause is IFAST in GStreamer jpegdec. /usr/include/jpeglib.h on this machine has JDCT_FASTEST = JDCT_IFAST and JDCT_DEFAULT = JDCT_ISLOW; that jpegdec uses JDCT_FASTEST as default is from knowledge of GStreamer. The higher error of the legacy picture (7.31 against 7.05) agrees with it. Whether IFAST gives a difference of 0 is not measured (I did not run tests, as instructed).

The proposed fix has the right direction but is not complete:
- An environment variable that is read at open cannot be changed by a JVM test. Five tests that compare with imread (colorJpegDecodesToExactLuma, chroma422JpegDecodesToExactLuma, colorJpegDecodesToExactBgr, grayscaleJpegGivesOneChannelInEachMode, jpegWithoutHuffmanTablesDecodesToTheSamePixels) fail with an IFAST default if they cannot select ISLOW.
- It does not cover BGR output and color cameras, where IFAST cannot give equality with base.
- Corrected method from the check (use THIS method unless the "Decisions" section says something different): Owner decision first: (A) make the direct decode equal to base (steps below), or (B) keep ISLOW and accept in writing that the direct capture gives a more accurate picture that is not equal to base (then only correct the commit/script text and keep the test as it is). The hard requirement asks for (A).

Steps for (A):

1. Native, /home/user/photonvision-mltag/tools/jetson-gstreamer-capture-jni/src/GStreamerCaptureJNI.cpp:
 - struct Capture: add std::atomic<int> jpegDctMethod.
 - createCapture(): set it from PV_GSTREAMER_CAPTURE_JPEG_IDCT. Value "islow" or "accurate" gives JDCT_ISLOW. Each other value and no value give JDCT_IFAST (the default of GStreamer jpegdec, thus the base behavior).
 - decodeJpeg(): add a parameter J_DCT_METHOD dctMethod and use it at line 1286 in place of JDCT_ISLOW. Correct the comment at line 1285. Keep do_fancy_upsampling as it is (it has no effect on luma).
 - frameFromJpegSample(): pass capture->jpegDctMethod in the two decodeJpeg calls (lines 1365 and 1379). This covers the first frame in grab() and the worker.
 - New JNI function setJpegIdctAccurate(jlong capturePtr, jboolean accurate) that stores JDCT_ISLOW or JDCT_IFAST in capture->jpegDctMethod. New capability bit (value 32) in getCapabilities().

2. Java: /home/user/photonvision-mltag/photon-core/src/main/java/org/photonvision/gstreamer/GStreamerCaptureJNI.java: add CAP_JPEG_IDCT = 32 and the native method. GStreamerCapture.java: add setJpegIdctAccurate(boolean) that calls it only when the capability is present. No change in GStreamerUSBFrameProvider.

3. Tests, /home/user/photonvision-mltag/photon-core/src/test/java/org/photonvision/gstreamer/GStreamerCaptureTest.java:
 - In the five tests that compare with imread, call capture.setJpegIdctAccurate(true) immediately after open and before the first grab. They then continue to prove the decode path (stride, Huffman tables, channel order) against the libjpeg reference.
 - Replace directGrayImageIsNotLessAccurateThanLegacyImage with three tests on the same mono 4:2:2 JPEG (1600x1304, quality 90):
   a. Default mode, OUTPUT_GRAY: assertSamePixels against "jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=GRAY8". This isolates the IDCT. Expected difference 0.
   b. Default mode, OUTPUT_GRAY: assertSamePixels against "jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=BGR" followed by Imgproc.cvtColor BGR2GRAY (the exact base path). Also OUTPUT_BGR against the BGR picture of that pipeline.
   c. Accurate mode: the old assertion directError <= legacyError.
 - Proof of the fault: tests a and b fail with the current code (largest difference 6) and must pass after the change.
 - Add the same comparison (a, b) for the real camera files in /home/user/photonvision-mltag/runs/jetson-frame-captures/ if they are original camera JPEG frames, because a real camera can have chroma that is not neutral.
 - Run the tests on the x86 host and on one ARM board. The SIMD code of libjpeg-turbo is different for each architecture, and the capture library must use the same libjpeg as jpegdec on that machine.

4. If test a passes but test b does not: the remaining difference comes from videoconvert (YUV to BGR rounding). The capture library cannot reproduce that. Report the measured largest difference to the owner. Exact equality is then only possible with PV_USB_GSTREAMER_CAPTURE_MODE=legacy.

5. Limits that IFAST does not remove; write them in the two run scripts (tools/rubik-run-photonvision-mltag-gstreamer.sh, tools/orangepi5-run-photonvision-mltag-gstreamer.sh) next to PV_USB_GSTREAMER_CPU_FORMAT and PV_USB_GSTREAMER_CAPTURE_MODE:
 - Color camera: GRAY8 mode uses the luma only, and direct BGR uses the libjpeg color conversion with fancy upsampling. Both differ from jpegdec + videoconvert. For data equal to base with a color camera use CAPTURE_MODE=legacy and CPU_FORMAT=BGR.
 - PV_USB_GSTREAMER_HW_JPEG_DECODER (off by default): a hardware decoder has its own IDCT, so its picture is not equal to base.
 - Document PV_GSTREAMER_CAPTURE_JPEG_IDCT=islow as the more accurate option that is not equal to base.

6. After the change run the JPEG stage of MLTagStageBenchmark (PV_MLTAG_BENCH=1) one time to confirm that the direct decode time is not longer with IFAST, because the FPS estimate uses that time. Run it when the machine is free of the timing measurement.
- Note of the lead engineer: Use the "Decisions" section for the default (accurate) and for the list of tests.

### F3: test that the tag decode and the pose estimation give the same data as the base commit
- Why: the tests of the new code compare the parallel code with the serial code of the SAME build. No test compares with the code before the speed work (base commit 412ccea). The owner needs a proof that tag IDs, corners, homography and poses are equal to the base for the same picture and the same regions.
- Method:
  1. Make a second git work tree of the base commit outside the project work tree: `git -C <scratch>/wt worktree add <scratch>/wt-base 412ccea` (detached). Do not change the branch wt-impl.
  2. In that base work tree add a temporary JUnit test class (it is NOT committed anywhere) that makes a golden file. Input pictures: the files test-resources/testimages/2024/*.jpg, test-resources/testimages/2026/*.jpg and test-resources/testimages/apriltag/36h11_stress_test.png, read with Imgcodecs.imread as BGR (IMREAD_COLOR). For each picture: (a) regions = the bounding boxes of the tags that a full-frame AprilTagDetector (family 36h11, the configuration and quad parameters of the base AprilTagROIDecodePipe.ROIDecodeParams) finds in the gray picture, each box made larger with the base AprilTagROIDecodePipe.expandBbox with padding 40, in the detector order; (b) the base AprilTagROIDecodePipe with its default parameters (ATR on, target dimension as the base default) and a second time with ATR off, run on the BGR picture with these regions; (c) for each detection the base single-tag pose estimation as the base AprilTagPipeline does it (AprilTagPoseEstimatorPipe with tag size of TargetModel.kAprilTag6p5in_36h11, numIterations 40, and a pinhole calibration for the picture size with an 80 degree horizontal field of view and zero distortion). Write for each picture and each ATR mode: the regions, and for each detection the ID, hamming, decision margin (Double.toString), the 4 corners, the center, the 9 homography values, and the pose estimate values (both transforms and both errors). Use a simple text or JSON format with Double.toString so that the values are exact.
  3. Run that test in the base work tree (the first gradle build there takes some minutes; use the same JDK option). If the base test resources or the base API differ from this text, adapt the temporary class to the base API and say what you did.
  4. Copy the golden file to the project work tree as test-resources/mltag/base-412ccea-roi-decode-pose.txt (new folder) and add the source text of the temporary generator class as a comment block or as a file test-resources/mltag/README.txt that says how the golden file was made (commit, inputs, parameters), so it can be made again.
  5. In the project work tree add the test class photon-core/src/test/java/org/photonvision/vision/pipe/BaseEqualityTest.java. For each picture and each ATR mode it gives the SAME regions (from the golden file) to the present AprilTagROIDecodePipe and the present pose code, in these variants: serial decode with the BGR picture, parallel decode with the BGR picture, parallel decode with the 1-channel gray picture (cvtColor BGR2GRAY of the same picture), and the pose estimation through the present parallel pose path. Each value must be EXACTLY equal to the golden value (compare the doubles with ==, or with Double.doubleToLongBits). Look at AprilTagROIParallelDecodeTest and AprilTagParallelPoseTest for how the present tests drive these pipes.
  6. If a variant is not exactly equal, do NOT change product code and do NOT weaken the test. Keep the failing assertion, and report the picture, the value, the two numbers and the variant. That is a result that the lead engineer needs.
  7. Remove the base work tree at the end: `git -C <scratch>/wt worktree remove --force <scratch>/wt-base`.
