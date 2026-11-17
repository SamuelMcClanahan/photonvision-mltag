/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.frame.provider;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.awt.Color;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.LockSupport;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.common.util.ColorHelper;
import org.photonvision.common.util.TestUtils;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.gstreamer.GStreamerCapture;
import org.photonvision.gstreamer.GStreamerCaptureJNI;
import org.photonvision.gstreamer.GStreamerFrame;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.calibration.JsonMatOfDouble;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameCaptureMetadata;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.objects.Letterbox;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.ObjectDetector;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.ImageRotationMode;
import org.photonvision.vision.opencv.OpenCVThreadSetting;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeParams;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;
import org.photonvision.vision.pipe.impl.VisionWorkerPool;
import org.photonvision.vision.pipeline.AprilTagPipeline;
import org.photonvision.vision.pipeline.AprilTagPipelineSettings;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TargetModel;
import org.photonvision.vision.target.TrackedTarget;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagDetector;

/**
 * End-to-end simulation of the ML AprilTag pipeline for a 1600x1304 60 FPS monochrome camera
 * (ThriftyCam) on a coprocessor that has four fast cores and an NPU.
 *
 * <p>What is real: the MJPEG frames, the native capture library and its JPEG decode thread, the
 * model input preparation, the AprilTag pipeline (ROI decode, pose estimation), and all threads.
 *
 * <p>PV_MLTAG_SIM_CAMERA_FORMAT selects the format of the simulated camera. "mjpeg" (default) is
 * Motion-JPEG. "gray" is the uncompressed 8-bit gray mode (V4L2 GREY) of the ThriftyCam with
 * firmware 5.10 or newer. In the gray mode each frame is a raw file of 1600x1304 bytes, the capture
 * library gets GRAY8 buffers, and there is no JPEG decode. The frame set then needs about 750 MB of
 * disk for 360 frames.
 *
 * <p>What is simulated: the camera (a paced file source) and the NPU (a wait of the measured
 * inference time, limited to the number of NPU cores of the board).
 *
 * <p>The camera gives a set of different frames, not one frame. The frames are made from real
 * pictures: the Motion-JPEG captures of the robot cameras in runs/jetson-frame-captures and the
 * field pictures in test-resources/testimages. PV_MLTAG_SIM_PICTURE_FOLDERS selects other folders,
 * for example the "images" folder of a YOLO dataset (the label files then give the regions of the
 * simulated model). Each real picture gives a sequence of camera frames: the picture gets the size
 * of the camera, the view moves across it from frame to frame, and the lighting of each sequence is
 * a small amount different (brightness, tone curve, a brightness gradient). The model also gives
 * some regions where no tag is, as a real model does at a low confidence threshold. Two cameras
 * start at different frames of the set.
 *
 * <p>Before the load runs, a calm run (slow camera, one frame at a time) records the tag data of
 * each frame, with the pose iteration count of each load run. Each result of a load run must be
 * equal to the recorded data of its frame: the ID, the corners and the pose of each tag. The
 * comparison starts with the first result, so it includes the warm-up, where the pipeline changes
 * its inference mode. Each result must also have the capture time and the sequence ID of its own
 * frame, and the capture times of the results must increase.
 *
 * <p>The CPU of a coprocessor is slower than the CPU of a development computer. The simulation
 * models a CPU that is k times slower with time scaling: the camera runs at 60 x k FPS and the NPU
 * wait is divided by k. If this computer keeps 60 x k FPS, then a CPU that is k times slower keeps
 * 60 FPS with the real NPU time. The printed FPS and latency values are converted back to the time
 * of the coprocessor. Costs that do not get smaller with k (thread wake-up, sleep accuracy) make
 * the simulation pessimistic.
 *
 * <p>PV_MLTAG_SIM_MODE=work is a second way to model the slower CPU. The camera and the NPU run in
 * real time, and each CPU stage does k times its work (the JPEG decode decodes extra rows, each tag
 * region is decoded more than one time, the pose estimation does more iterations). In this mode the
 * thread wake-up times of the operating system have their real size, so it is the better mode for
 * the variation of the frame times (p99).
 *
 * <p>PV_MLTAG_SIM_SLOWDOWNS is a list of values for k (default 1.3,1.5,2.0). The single-thread
 * 7-Zip benchmark gives a ratio of 1.1 to 1.3 between a Cortex-A76 core of the RK3588 and a core of
 * the computer where this simulation was made, so 1.3 is the expected value, and 1.5 and 2.0 are
 * values with a reserve. Measure the ratio again on a different computer.
 *
 * <p>The JPEG decode is slower on the RK3588 than this general ratio. The benchmark pts/tjbench of
 * the Phoronix Test Suite (libjpeg-turbo, picture jpeg-test-1.JPG, 4:2:2 to BGR) gives 156
 * megapixels per second on the computer where this simulation was made. It gives 97.4 megapixels
 * per second on the Cortex-A76 of the RK3588 (openbenchmarking.org result for a Mixtile Blade 3).
 * So the JPEG decode is 1.6 times slower on the RK3588. PV_MLTAG_SIM_JPEG_FACTOR sets this factor
 * for the JPEG decode in the work mode with the mjpeg format. Without it, the JPEG decode uses the
 * value of k. The time scaling mode does not use it, because there all work has the same factor k.
 *
 * <p>The run scripts set PV_OPENCV_THREADS=1 and PV_GSTREAMER_CAPTURE_RT_PRIORITY=10. Use the same
 * settings for the simulation. The real-time priority needs root or CAP_SYS_NICE. If the kernel
 * refuses it, the simulation fails.
 *
 * <p>The test JVM must have exactly the CPUs of the board (4), because the number of CPUs changes
 * the number of parallel groups of the tag decode and the pose estimation, and the competition for
 * the CPUs. The simulation fails if the count is different. Set PV_MLTAG_SIM_ALLOW_CPU_COUNT=1 to
 * run anyway: then each report line says that it is not valid for the board. The run scripts use
 * "taskset -c 4-7", so on a computer with more CPUs start Gradle with taskset and with no daemon. A
 * Gradle daemon that already runs keeps its own CPU mask, and the test JVM gets that mask.
 * PV_MLTAG_SIM_APP_CPUS=2,3 gives the CPU mask only to the threads of the application. It does not
 * move the GC and JIT threads of the JVM. taskset on the JVM moves all threads.
 *
 * <p>The report gives for each camera: the mean result rate (resultFps), the rate that 99% of the
 * times between two results are better than (p99FrameFps), the lowest value in 100 of the FPS value
 * that PhotonVision shows, which is a mean of 20 frames (p99ReportedFps), and the lowest rate in a
 * window of one second (worstSecondFps).
 *
 * <pre>
 * ./gradlew --stop
 * PV_MLTAG_SIM=1 PV_MLTAG_SIM_MODE=work PV_MLTAG_SIM_CAMERAS=2 PV_MLTAG_SIM_SECONDS=30 \
 *   PV_OPENCV_THREADS=1 PV_GSTREAMER_CAPTURE_RT_PRIORITY=10 \
 *   PV_GSTREAMER_CAPTURE_JNI_LIB=/path/libpv_gstreamer_capture_jni.so \
 *   taskset -c 0-3 ./gradlew --no-daemon :photon-core:test \
 *   --tests "*MLTagEndToEndSimulation.orangePi5"
 * </pre>
 *
 * <p>Use 4 CPUs on 4 different physical cores for taskset (see lscpu -e).
 */
public class MLTagEndToEndSimulation {
    private static final int WIDTH = 1600;
    private static final int HEIGHT = 1304;
    private static final double CAMERA_FPS = 60.0;
    private static final int ROI_PADDING = 40;

    /** PV_MLTAG_SIM_MODE=work: real time and more CPU work. Default: time scaling. */
    private static final boolean WORK_MODE = "work".equals(System.getenv("PV_MLTAG_SIM_MODE"));

    /** PV_MLTAG_SIM_ATR=0: adaptive tag resizing off. Default: on, as in a new pipeline. */
    private static final boolean ATR_ENABLED = !"0".equals(System.getenv("PV_MLTAG_SIM_ATR"));

    /** PV_MLTAG_SIM_NULL=1: the vision loops take the frames and do no work (reference). */
    private static final boolean NULL_PIPELINE = "1".equals(System.getenv("PV_MLTAG_SIM_NULL"));

    /**
     * PV_MLTAG_SIM_CAMERA_FORMAT: "mjpeg" (default) or "gray". The gray mode simulates the
     * uncompressed GRAY8 mode of the camera, where the coprocessor does no JPEG decode.
     */
    private static final String CAMERA_FORMAT = cameraFormat();

    private static final boolean GRAY_CAMERA = "gray".equals(CAMERA_FORMAT);

    private static String cameraFormat() {
        String raw = System.getenv("PV_MLTAG_SIM_CAMERA_FORMAT");
        return raw == null || raw.isBlank() ? "mjpeg" : raw.trim().toLowerCase(Locale.ROOT);
    }

    /**
     * PV_MLTAG_SIM_JPEG_FACTOR: the CPU factor of the JPEG decode in the work mode, or null if the
     * setting is absent. It is separate from k because the JPEG decode of the RK3588 is slower than
     * its general CPU speed (see the class comment). The time scaling mode does not use it.
     */
    private static final Double JPEG_FACTOR = jpegFactorSetting();

    private static Double jpegFactorSetting() {
        String raw = System.getenv("PV_MLTAG_SIM_JPEG_FACTOR");
        return raw == null || raw.isBlank() ? null : Double.parseDouble(raw.trim());
    }

    /** The factor of the JPEG decode work for a scenario with the general CPU factor k. */
    private static double jpegFactor(double slowdown) {
        return JPEG_FACTOR != null ? JPEG_FACTOR : slowdown;
    }

    /**
     * The extra JPEG decode work in percent for a scenario. Only the work mode with the mjpeg format
     * adds work. In the gray mode there is no JPEG decode, so the value is 0.
     */
    private static int extraDecodePercent(double slowdown) {
        if (!WORK_MODE || GRAY_CAMERA) {
            return 0;
        }
        return (int) Math.round((jpegFactor(slowdown) - 1.0) * 100);
    }

    @TempDir static Path tempDir;

    /**
     * A coprocessor profile.
     *
     * @param name Name for the report.
     * @param npuCores Number of inferences that the NPU can run at the same time.
     * @param inferenceMillis Mean time of the detect call of the NPU library for one frame.
     * @param inferenceSpread Standard deviation of the inference time, as a part of the mean.
     * @param lanes Value of PV_MLTAG_INFERENCE_LANES in the run script of the board.
     * @param cpus Number of CPUs that the run script gives the JVM (taskset).
     */
    private record Board(
            String name,
            int npuCores,
            double inferenceMillis,
            double inferenceSpread,
            int lanes,
            int cpus) {}

    /**
     * Rubik Pi 3 (QCS6490). The AprilTag benchmark of 2026-07-09 measured 5.1 ms for the complete
     * model stage, including about 0.5 ms of input preparation that the simulation does itself.
     * Qualcomm AI Hub gives 3.7 to 4.1 ms for YOLOv11 detection (w8a8) on this NPU. One HTP runs one
     * graph at a time.
     */
    private static final Board RUBIK = new Board("Rubik Pi 3", 1, 4.6, 0.0, 1, 4);

    /**
     * Orange Pi 5 (RK3588). The RKNN benchmark of this repository
     * (docs/source/docs/benchmarks/rknn-model-benchmarks.md) measured 22,989 +/- 2,356 microseconds
     * for one YOLOv11 inference (int8, 640x640) on an Orange Pi 5, as the mean and the standard
     * deviation of 5,000 inferences on one NPU core. That measurement had the operator timing on,
     * which makes an inference slower: other measurements without it give 16.7 ms (Seeed) and 18.7 ms
     * (Radxa). The simulation uses the larger value and its spread. The NPU has three cores. When
     * three inferences run at the same time, each one can be slower. A measurement with three YOLOv5s
     * instances gave 18.8 ms for each instance. PV_MLTAG_SIM_INFERENCE_MS is a list of mean inference
     * times to find the effect of that.
     */
    private static final Board ORANGE_PI = new Board("Orange Pi 5", 3, 23.0, 0.1025, 2, 4);

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
        // VisionRunner does this in the application. The run scripts set PV_OPENCV_THREADS=1.
        OpenCVThreadSetting.apply();
    }

    private static boolean enabled() {
        String raw = System.getenv("PV_MLTAG_SIM");
        return raw != null && !raw.isBlank() && !raw.equals("0");
    }

    private static void line(String format, Object... args) {
        System.out.println("[sim] " + String.format(Locale.ROOT, format, args));
    }

    // ---------------------------------------------------------------------------------------
    // Frame set
    // ---------------------------------------------------------------------------------------

    /**
     * What the simulation knows about one camera frame.
     *
     * @param source Index into {@link #sourceNames}.
     * @param frameBytes Size of the frame file in the active camera format.
     * @param boxes The regions that the simulated model gives for this frame.
     */
    private record SceneFrame(int source, long frameBytes, List<RotatedRect> boxes) {}

    private static SceneFrame[] sceneFrames;
    private static List<String> sourceNames;
    private static Path frameDirectory;

    /** The pose iteration count of a pipeline with the real CPU work (k = 1). */
    private static final int BASE_ITERATIONS = 40;

    /**
     * The pose iteration count for a work multiplier. The work mode makes the pose estimation do more
     * iterations. makeSettings and the reference use this one function, so the two cannot differ.
     */
    private static int iterationsFor(double workMultiplier) {
        return (int) Math.round(BASE_ITERATIONS * workMultiplier);
    }

    /**
     * For each pose iteration count: the tag data of each frame from a calm run (a slow camera and a
     * pipeline that completes each frame before the next frame arrives). Each result of a load run
     * must be equal to the data of its iteration count. The pose depends on the iteration count, so
     * one reference for all counts would make it necessary to skip the pose.
     */
    private static final Map<Integer, List<List<double[]>>> referenceByIterations = new HashMap<>();

    /** The CPU count of the JVM before the simulation sets a CPU mask. -1 before the first check. */
    private static int hostCpus = -1;

    /** The CPU count of the application threads: hostCpus, or the CPUs of PV_MLTAG_SIM_APP_CPUS. */
    private static int appCpus;

    /** True if PV_MLTAG_SIM_ALLOW_CPU_COUNT=1 let a run continue with the wrong CPU count. */
    private static boolean cpuCountNotValid;

    /**
     * The extra JPEG decode percent that the simulation gave the native library last. The library
     * has no getter, and the value is one setting for the whole process. A test reads this copy to
     * see that a scenario puts the value back to 0, also after a failure.
     */
    private static volatile int activeExtraDecodePercent;

    /** The captures of the last scenario. A test reads it to see that each one was closed. */
    private static volatile List<GStreamerCapture> lastScenarioCaptures = List.of();

    private static void setExtraDecodePercent(int percent) {
        GStreamerCaptureJNI.setTestExtraDecodePercent(percent);
        activeExtraDecodePercent = percent;
    }

    /** Number of frames that the simulated model could not identify. It must stay 0. */
    private static final AtomicInteger unknownFrames = new AtomicInteger();

    /**
     * The model input path of the last simulated detect call. The report prints it, so an FPS
     * estimate says if it used the monochrome fast path or the slower base operation.
     */
    private static volatile String modelInputPath = "none";

    // The index of a frame is in the top left corner of the picture: 12 blocks of 16x16 pixels
    // (black = 0, white = 1) and below them the inverse blocks. No region is near these blocks.
    private static final int INDEX_BITS = 12;
    private static final int INDEX_BLOCK = 16;

    private static void drawFrameIndex(Mat scene, int index) {
        for (int bit = 0; bit < INDEX_BITS; bit++) {
            boolean one = ((index >> bit) & 1) != 0;
            int x = bit * INDEX_BLOCK;
            Imgproc.rectangle(
                    scene,
                    new Point(x, 0),
                    new Point(x + INDEX_BLOCK - 1, INDEX_BLOCK - 1),
                    new Scalar(one ? 255 : 0),
                    -1);
            Imgproc.rectangle(
                    scene,
                    new Point(x, INDEX_BLOCK),
                    new Point(x + INDEX_BLOCK - 1, 2 * INDEX_BLOCK - 1),
                    new Scalar(one ? 0 : 255),
                    -1);
        }
    }

    /** Returns the index of a decoded frame, or -1 if the picture has no valid index. */
    private static int readFrameIndex(Mat image) {
        if (image.empty()
                || image.rows() < 2 * INDEX_BLOCK
                || image.cols() < INDEX_BITS * INDEX_BLOCK) {
            return -1;
        }
        int channels = image.channels();
        byte[] upper = new byte[INDEX_BITS * INDEX_BLOCK * channels];
        byte[] lower = new byte[INDEX_BITS * INDEX_BLOCK * channels];
        image.get(INDEX_BLOCK / 2, 0, upper);
        image.get(INDEX_BLOCK + INDEX_BLOCK / 2, 0, lower);
        int index = 0;
        for (int bit = 0; bit < INDEX_BITS; bit++) {
            int offset = (bit * INDEX_BLOCK + INDEX_BLOCK / 2) * channels;
            boolean one = (upper[offset] & 0xFF) > 128;
            boolean inverse = (lower[offset] & 0xFF) > 128;
            if (one == inverse) {
                return -1;
            }
            if (one) {
                index |= 1 << bit;
            }
        }
        return sceneFrames != null && index < sceneFrames.length ? index : -1;
    }

    private static RotatedRect boundingBox(double[] corners) {
        double minX = Double.MAX_VALUE;
        double maxX = -Double.MAX_VALUE;
        double minY = Double.MAX_VALUE;
        double maxY = -Double.MAX_VALUE;
        for (int i = 0; i < 4; i++) {
            minX = Math.min(minX, corners[i * 2]);
            maxX = Math.max(maxX, corners[i * 2]);
            minY = Math.min(minY, corners[i * 2 + 1]);
            maxY = Math.max(maxY, corners[i * 2 + 1]);
        }
        return new RotatedRect(
                new Point((minX + maxX) / 2.0, (minY + maxY) / 2.0), new Size(maxX - minX, maxY - minY), 0);
    }

    private static boolean centerIsInPicture(RotatedRect box) {
        // Not in the top left corner, where the frame index is.
        boolean inIndexArea =
                box.center.x - box.size.width / 2.0 < INDEX_BITS * INDEX_BLOCK + 60
                        && box.center.y - box.size.height / 2.0 < 2 * INDEX_BLOCK + 60;
        return !inIndexArea
                && box.center.x > 8
                && box.center.x < WIDTH - 8
                && box.center.y > 8
                && box.center.y < HEIGHT - 8;
    }

    private static boolean boxesAreNear(RotatedRect a, RotatedRect b, double margin) {
        return Math.abs(a.center.x - b.center.x) < (a.size.width + b.size.width) / 2.0 + margin
                && Math.abs(a.center.y - b.center.y) < (a.size.height + b.size.height) / 2.0 + margin;
    }

    /** A model does not give the exact box of a tag. This changes a box by a few percent. */
    private static RotatedRect modelBox(RotatedRect box, Random random) {
        double scale = 0.97 + random.nextDouble() * 0.13;
        return new RotatedRect(
                new Point(
                        box.center.x + (random.nextDouble() - 0.5) * 0.06 * box.size.width,
                        box.center.y + (random.nextDouble() - 0.5) * 0.06 * box.size.height),
                new Size(box.size.width * scale, box.size.height * scale),
                0);
    }

    /** The boxes of the tags that the AprilTag detector finds in a full picture. */
    private static List<RotatedRect> findTags(Mat gray, AprilTagDetector detector) {
        List<RotatedRect> boxes = new ArrayList<>();
        for (AprilTagDetection detection : detector.detect(gray)) {
            if (detection.getHamming() > 0 || detection.getDecisionMargin() < 35) {
                continue;
            }
            double[] corners = new double[8];
            for (int i = 0; i < 4; i++) {
                corners[i * 2] = detection.getCornerX(i);
                corners[i * 2 + 1] = detection.getCornerY(i);
            }
            boxes.add(boundingBox(corners));
        }
        return boxes;
    }

    private static RotatedRect shifted(RotatedRect box, double dx, double dy) {
        return new RotatedRect(
                new Point(box.center.x - dx, box.center.y - dy), box.size.clone(), box.angle);
    }

    /**
     * The pictures of one source: one real picture, or the frames of one real Motion-JPEG capture.
     * The pictures are gray and have the scale of the simulated camera.
     */
    private static final class Source {
        final String name;
        final List<Mat> pictures = new ArrayList<>();

        /** The regions with a tag, in the coordinates of the scaled pictures. */
        List<RotatedRect> tagRegions = new ArrayList<>();

        /** True: the regions are from a YOLO label file. False: from the AprilTag detector. */
        boolean regionsFromLabels = false;

        Source(String name) {
            this.name = name;
        }
    }

    /** The folders in the repository that contain real pictures with AprilTags. */
    private static List<Path> defaultPictureFolders() {
        Path testImages = TestUtils.getTestImagesPath(false).toAbsolutePath().normalize();
        Path repository = testImages.getParent().getParent();
        // The folder "testimages/apriltag" is not in this list: it contains test sheets for the
        // detector (for example one picture with 125 tags), not views of a camera.
        return List.of(
                // Frames from the cameras of the robot (Motion-JPEG captures).
                repository.resolve("runs").resolve("jetson-frame-captures"),
                // Field pictures.
                testImages.resolve("2024"),
                testImages.resolve("2026"));
    }

    /** Cuts a Motion-JPEG capture into its JPEG frames. */
    private static List<byte[]> splitMotionJpeg(byte[] data) {
        List<Integer> starts = new ArrayList<>();
        for (int i = 0; i + 2 < data.length; i++) {
            if ((data[i] & 0xFF) == 0xFF
                    && (data[i + 1] & 0xFF) == 0xD8
                    && (data[i + 2] & 0xFF) == 0xFF) {
                starts.add(i);
            }
        }
        List<byte[]> frames = new ArrayList<>();
        for (int i = 0; i < starts.size(); i++) {
            int end = i + 1 < starts.size() ? starts.get(i + 1) : data.length;
            frames.add(Arrays.copyOfRange(data, starts.get(i), end));
        }
        return frames;
    }

    /**
     * Makes a picture as large as the picture of the simulated camera, with room to move the view.
     */
    private static Mat scaleToCamera(Mat picture) {
        double scale =
                Math.max((double) WIDTH / picture.cols(), (double) HEIGHT / picture.rows()) * 1.06;
        Mat scaled = new Mat();
        Imgproc.resize(
                picture,
                scaled,
                new Size(
                        Math.max(WIDTH, Math.round(picture.cols() * scale)),
                        Math.max(HEIGHT, Math.round(picture.rows() * scale))),
                0,
                0,
                scale > 1.0 ? Imgproc.INTER_CUBIC : Imgproc.INTER_AREA);
        return scaled;
    }

    /**
     * Reads the YOLO label file of a dataset picture, if there is one: "labels/name.txt" next to the
     * "images" folder, or "name.txt" next to the picture. Each line is: class, center x, center y,
     * width, height (parts of the picture size). Returns null if there is no label file.
     */
    private static List<RotatedRect> readLabels(Path picture, int scaledWidth, int scaledHeight)
            throws java.io.IOException {
        String fileName = picture.getFileName().toString();
        String base =
                fileName.contains(".") ? fileName.substring(0, fileName.lastIndexOf('.')) : fileName;
        Path folder = picture.toAbsolutePath().getParent();
        List<Path> candidates = new ArrayList<>();
        if (folder.getParent() != null) {
            candidates.add(folder.getParent().resolve("labels").resolve(base + ".txt"));
        }
        candidates.add(folder.resolve(base + ".txt"));
        for (Path labels : candidates) {
            if (!java.nio.file.Files.isRegularFile(labels)) {
                continue;
            }
            List<RotatedRect> boxes = new ArrayList<>();
            for (String line : java.nio.file.Files.readAllLines(labels)) {
                String[] parts = line.trim().split("\\s+");
                if (parts.length < 5) {
                    continue;
                }
                boxes.add(
                        new RotatedRect(
                                new Point(
                                        Double.parseDouble(parts[1]) * scaledWidth,
                                        Double.parseDouble(parts[2]) * scaledHeight),
                                new Size(
                                        Double.parseDouble(parts[3]) * scaledWidth,
                                        Double.parseDouble(parts[4]) * scaledHeight),
                                0));
            }
            return boxes;
        }
        return null;
    }

    /**
     * Reads the real pictures. PV_MLTAG_SIM_PICTURE_FOLDERS is a list of folders, with ":" between
     * them (for example the "images" folder of a YOLO dataset). The default is the folders of the
     * repository that contain real pictures with AprilTags.
     */
    private static List<Source> readSources(AprilTagDetector detector) throws java.io.IOException {
        List<Path> folders = new ArrayList<>();
        String setting = System.getenv("PV_MLTAG_SIM_PICTURE_FOLDERS");
        if (setting != null && !setting.isBlank()) {
            for (String folder : setting.split(":")) {
                folders.add(Path.of(folder.trim()));
            }
        } else {
            folders.addAll(defaultPictureFolders());
        }
        int maxSources = intList("PV_MLTAG_SIM_MAX_PICTURES", new int[] {60})[0];

        List<Source> sources = new ArrayList<>();
        for (Path folder : folders) {
            if (!java.nio.file.Files.isDirectory(folder)) {
                line("picture folder not found: %s", folder);
                continue;
            }
            List<Path> files = new ArrayList<>();
            try (var stream = java.nio.file.Files.list(folder)) {
                stream.sorted().forEach(files::add);
            }
            for (Path file : files) {
                if (sources.size() >= maxSources) {
                    break;
                }
                String lower = file.getFileName().toString().toLowerCase(Locale.ROOT);
                String label = folder.getFileName() + "/" + file.getFileName();
                Source source = new Source(label);
                Path labelPicture = null;
                if (lower.endsWith(".mjpg") || lower.endsWith(".mjpeg")) {
                    for (byte[] jpeg : splitMotionJpeg(java.nio.file.Files.readAllBytes(file))) {
                        MatOfByte bytes = new MatOfByte(jpeg);
                        Mat picture = Imgcodecs.imdecode(bytes, Imgcodecs.IMREAD_GRAYSCALE);
                        bytes.release();
                        if (!picture.empty()) {
                            source.pictures.add(picture);
                        }
                    }
                } else if (lower.endsWith(".jpg") || lower.endsWith(".jpeg") || lower.endsWith(".png")) {
                    // A ".jpg" file with the same name as a ".mjpg" capture is a copy of the capture.
                    String base = file.toString().substring(0, file.toString().lastIndexOf('.'));
                    if (java.nio.file.Files.exists(Path.of(base + ".mjpg"))) {
                        continue;
                    }
                    Mat picture = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_GRAYSCALE);
                    if (!picture.empty()) {
                        source.pictures.add(picture);
                        labelPicture = file;
                    }
                }
                // A very small picture is not a camera picture.
                if (source.pictures.isEmpty()
                        || source.pictures.get(0).cols() < 600
                        || source.pictures.get(0).rows() < 400) {
                    for (Mat picture : source.pictures) {
                        picture.release();
                    }
                    continue;
                }
                for (int i = 0; i < source.pictures.size(); i++) {
                    Mat scaled = scaleToCamera(source.pictures.get(i));
                    source.pictures.get(i).release();
                    source.pictures.set(i, scaled);
                }
                Mat first = source.pictures.get(0);
                List<RotatedRect> labels =
                        labelPicture == null ? null : readLabels(labelPicture, first.cols(), first.rows());
                if (labels != null) {
                    source.tagRegions = labels;
                    source.regionsFromLabels = true;
                } else {
                    source.tagRegions = findTags(first, detector);
                }
                sources.add(source);
            }
        }
        return sources;
    }

    /**
     * A small change of the lighting for one sequence of frames.
     *
     * @param gain Brightness factor.
     * @param gamma Exponent of the tone curve (1 = no change).
     * @param offset Value that is added to each pixel, in gray levels.
     * @param gradient One side of the picture is this part brighter than the center, and the other
     *     side is this part darker.
     * @param gradientAngle Direction of the brightness gradient, in radians.
     */
    private record Lighting(
            double gain, double gamma, double offset, double gradient, double gradientAngle) {}

    /** A map that makes one side of the picture brighter and the other side darker. */
    private static Mat makeGradientMap(Lighting lighting) {
        Mat map = new Mat(HEIGHT, WIDTH, CvType.CV_32FC1);
        float[] row = new float[WIDTH];
        double cos = Math.cos(lighting.gradientAngle());
        double sin = Math.sin(lighting.gradientAngle());
        for (int y = 0; y < HEIGHT; y++) {
            double v = (y - HEIGHT / 2.0) / (HEIGHT / 2.0);
            for (int x = 0; x < WIDTH; x++) {
                double u = (x - WIDTH / 2.0) / (WIDTH / 2.0);
                row[x] = (float) (1.0 + lighting.gradient() * (u * cos + v * sin));
            }
            map.put(y, 0, row);
        }
        return map;
    }

    /**
     * Changes the lighting of a picture a small amount (tone curve, brightness, gradient, offset) and
     * adds the low noise of a camera sensor at full resolution: a picture that was made larger has
     * less noise than a real frame of the camera.
     *
     * @param drift The brightness change of this frame in its sequence (for example 0.02 = 2%).
     */
    private static void applyLighting(Mat scene, Lighting lighting, Mat gradientMap, double drift) {
        Mat signal = new Mat();
        scene.convertTo(signal, CvType.CV_32F, 1.0 / 255.0);
        Core.pow(signal, lighting.gamma(), signal);
        Core.multiply(signal, gradientMap, signal, 255.0 * lighting.gain() * (1.0 + drift));
        Core.add(signal, new Scalar(lighting.offset()), signal);
        Core.max(signal, new Scalar(0), signal);

        // Standard deviation of the noise = square root of (read noise squared + shot noise x signal).
        final double readNoise = 1.2;
        final double shotNoise = 0.02;
        Mat sigma = new Mat();
        Core.multiply(signal, new Scalar(shotNoise), sigma);
        Core.add(sigma, new Scalar(readNoise * readNoise), sigma);
        Core.sqrt(sigma, sigma);
        Mat noise = new Mat(scene.rows(), scene.cols(), CvType.CV_32FC1);
        Core.randn(noise, 0, 1);
        Core.multiply(noise, sigma, noise);
        Core.add(signal, noise, signal);
        signal.convertTo(scene, CvType.CV_8U);
        signal.release();
        sigma.release();
        noise.release();
    }

    /**
     * The largest Motion-JPEG frame of the simulated camera, and the JPEG quality. The frame size of
     * the real camera is not known. The Motion-JPEG captures of the robot cameras in
     * runs/jetson-frame-captures (1280x800) have 62 to 78 kB for each frame, which is 61 to 76 kB for
     * each megapixel. The defaults here give 181 to 349 kB for each 1600x1304 frame, which is 87 to
     * 167 kB for each megapixel: about two times the data of the real captures, so the JPEG decode
     * work of the simulation is on the high side. PV_MLTAG_SIM_MAX_JPEG_KB and
     * PV_MLTAG_SIM_JPEG_QUALITY change the values.
     */
    private static final int MAX_JPEG_BYTES =
            intList("PV_MLTAG_SIM_MAX_JPEG_KB", new int[] {350})[0] * 1000;

    private static final int JPEG_QUALITY = intList("PV_MLTAG_SIM_JPEG_QUALITY", new int[] {90})[0];

    /**
     * Writes a frame as a monochrome UVC camera sends it: YCbCr 4:2:2 Motion-JPEG with neutral
     * chroma. For a frame that would be larger than the limit, the quality is decreased, as the rate
     * control of a camera does it. Returns the size of the frame.
     */
    private static long writeJpeg(Mat scene, Path file) throws java.io.IOException {
        Mat bgr = new Mat();
        Imgproc.cvtColor(scene, bgr, Imgproc.COLOR_GRAY2BGR);
        MatOfByte encoded = new MatOfByte();
        try {
            for (int quality = JPEG_QUALITY; quality >= 30; quality -= 5) {
                assertTrue(
                        Imgcodecs.imencode(
                                ".jpg",
                                bgr,
                                encoded,
                                new MatOfInt(
                                        Imgcodecs.IMWRITE_JPEG_QUALITY,
                                        quality,
                                        Imgcodecs.IMWRITE_JPEG_SAMPLING_FACTOR,
                                        Imgcodecs.IMWRITE_JPEG_SAMPLING_FACTOR_422)));
                if (encoded.total() <= MAX_JPEG_BYTES) {
                    break;
                }
            }
            byte[] bytes = encoded.toArray();
            java.nio.file.Files.write(file, bytes);
            return bytes.length;
        } finally {
            bgr.release();
            encoded.release();
        }
    }

    /**
     * Writes a frame as the gray mode of the camera sends it: the 8-bit picture, row after row, with
     * no header and no padding. 1600 is a multiple of 4, so the row stride of GRAY8 in GStreamer is
     * the width and the file has exactly the size of one buffer. Returns the size of the frame.
     */
    private static long writeGray(Mat scene, Path file) throws java.io.IOException {
        assertEquals(CvType.CV_8UC1, scene.type());
        assertEquals(WIDTH, scene.cols());
        assertEquals(HEIGHT, scene.rows());
        byte[] bytes = new byte[WIDTH * HEIGHT];
        // get() copies row after row, also when the Mat is not continuous.
        scene.get(0, 0, bytes);
        java.nio.file.Files.write(file, bytes);
        return bytes.length;
    }

    /** The file name pattern of the frames, for the active camera format. */
    private static String framePattern() {
        return GRAY_CAMERA ? "frame_%04d.gray" : "frame_%04d.jpg";
    }

    private static Path framePath(int index) {
        return frameDirectory.resolve(String.format(Locale.ROOT, framePattern(), index));
    }

    /** For each source in sequence: the number of regions without a tag that the model gives. */
    private static final int[] REGIONS_WITHOUT_TAG = {0, 0, 1, 0, 2, 0, 1, 3};

    /**
     * Writes the sequence of camera frames for one source. The view moves across the picture from
     * frame to frame, and the lighting of each sequence is a small amount different.
     */
    private static void renderSource(
            int sourceIndex, Source source, int frameCount, List<SceneFrame> frames)
            throws java.io.IOException {
        Random random = new Random(7919L * (sourceIndex + 1));
        Mat first = source.pictures.get(0);
        int rangeX = first.cols() - WIDTH;
        int rangeY = first.rows() - HEIGHT;

        // The view moves on a line from one position to a different position.
        double startX = random.nextDouble() * rangeX;
        double startY = random.nextDouble() * rangeY;
        double endX = random.nextDouble() * rangeX;
        double endY = random.nextDouble() * rangeY;

        Lighting lighting =
                new Lighting(
                        0.88 + random.nextDouble() * 0.24,
                        0.88 + random.nextDouble() * 0.26,
                        -6.0 + random.nextDouble() * 12.0,
                        random.nextDouble() * 0.08,
                        random.nextDouble() * 2.0 * Math.PI);
        Mat gradientMap = makeGradientMap(lighting);
        double driftPhase = random.nextDouble() * 2.0 * Math.PI;

        // Regions without a tag, as a model gives them at a low confidence threshold. They are
        // parts of the picture, so they move with the picture.
        List<RotatedRect> regionsWithoutTag = new ArrayList<>();
        int wanted = REGIONS_WITHOUT_TAG[sourceIndex % REGIONS_WITHOUT_TAG.length];
        for (int attempt = 0; attempt < 400 && regionsWithoutTag.size() < wanted; attempt++) {
            double edge = 70 + random.nextDouble() * 190;
            RotatedRect box =
                    new RotatedRect(
                            new Point(
                                    (startX + endX) / 2.0 + 300 + random.nextDouble() * (WIDTH - 500),
                                    (startY + endY) / 2.0 + 300 + random.nextDouble() * (HEIGHT - 500)),
                            new Size(edge, edge * (0.8 + random.nextDouble() * 0.4)),
                            0);
            boolean free = true;
            for (RotatedRect other : source.tagRegions) {
                free &= !boxesAreNear(box, other, 80);
            }
            for (RotatedRect other : regionsWithoutTag) {
                free &= !boxesAreNear(box, other, 40);
            }
            if (free) {
                regionsWithoutTag.add(box);
            }
        }

        for (int i = 0; i < frameCount; i++) {
            int index = frames.size();
            double t = frameCount > 1 ? i / (double) (frameCount - 1) : 0.0;
            int left = (int) Math.round(startX + (endX - startX) * t);
            int top = (int) Math.round(startY + (endY - startY) * t);
            Mat picture = source.pictures.get(i % source.pictures.size());
            Mat scene = picture.submat(new Rect(left, top, WIDTH, HEIGHT)).clone();

            List<RotatedRect> boxes = new ArrayList<>();
            for (RotatedRect box : source.tagRegions) {
                RotatedRect inPicture = shifted(box, left, top);
                if (centerIsInPicture(inPicture)) {
                    boxes.add(modelBox(inPicture, random));
                }
            }
            for (RotatedRect box : regionsWithoutTag) {
                RotatedRect inPicture = shifted(box, left, top);
                if (centerIsInPicture(inPicture)) {
                    boxes.add(inPicture);
                }
            }

            applyLighting(scene, lighting, gradientMap, 0.03 * Math.sin(driftPhase + i * 0.6));
            drawFrameIndex(scene, index);
            long bytes =
                    GRAY_CAMERA ? writeGray(scene, framePath(index)) : writeJpeg(scene, framePath(index));
            frames.add(new SceneFrame(sourceIndex, bytes, boxes));
            scene.release();
        }
        gradientMap.release();
    }

    /**
     * Makes the frame set from real pictures. PV_MLTAG_SIM_SCENE_FRAMES is the number of camera
     * frames for each picture (default 20). PV_MLTAG_SIM_SCENES is a list of source numbers (default:
     * all sources).
     */
    private static synchronized void prepareScene() throws Exception {
        if (sceneFrames != null) {
            return;
        }
        long start = System.nanoTime();
        // PV_MLTAG_SIM_FRAME_DIR keeps the frames in this directory, to look at them.
        String keepDirectory = System.getenv("PV_MLTAG_SIM_FRAME_DIR");
        frameDirectory =
                keepDirectory == null || keepDirectory.isBlank() ? tempDir : Path.of(keepDirectory);
        java.nio.file.Files.createDirectories(frameDirectory);
        Core.setRNGSeed(20261116);

        AprilTagDetector detector = new AprilTagDetector();
        detector.addFamily(AprilTagFamily.kTag36h11.getNativeName());
        ROIDecodeParams defaults = new ROIDecodeParams();
        detector.setConfig(defaults.detectorConfig);
        detector.setQuadThresholdParameters(defaults.quadParams);
        List<Source> allSources = readSources(detector);
        detector.close();
        assertTrue(allSources.size() >= 2, "The simulation needs two or more real pictures");

        int[] selection = new int[allSources.size()];
        for (int i = 0; i < selection.length; i++) {
            selection[i] = i;
        }
        selection = intList("PV_MLTAG_SIM_SCENES", selection);
        int framesForEachSource = intList("PV_MLTAG_SIM_SCENE_FRAMES", new int[] {20})[0];
        assertTrue(selection.length * framesForEachSource < (1 << INDEX_BITS), "Too many frames");

        List<String> names = new ArrayList<>();
        List<Boolean> fromLabels = new ArrayList<>();
        List<SceneFrame> frames = new ArrayList<>();
        for (int sourceIndex : selection) {
            Source source = allSources.get(sourceIndex);
            renderSource(names.size(), source, framesForEachSource, frames);
            names.add(source.name);
            fromLabels.add(source.regionsFromLabels);
        }
        for (Source source : allSources) {
            for (Mat picture : source.pictures) {
                picture.release();
            }
        }
        sourceNames = names;
        sceneFrames = frames.toArray(new SceneFrame[0]);

        // The tag count does not depend on the pose iterations, so the statistics use the base
        // reference.
        List<List<double[]>> baseReference = prepareReference(BASE_ITERATIONS);

        // What the frame set contains.
        long minBytes = Long.MAX_VALUE;
        long maxBytes = 0;
        long sumBytes = 0;
        int minBoxes = Integer.MAX_VALUE;
        int maxBoxes = 0;
        int sumBoxes = 0;
        int minTags = Integer.MAX_VALUE;
        int maxTags = 0;
        int sumTags = 0;
        for (int i = 0; i < sceneFrames.length; i++) {
            SceneFrame frame = sceneFrames[i];
            minBytes = Math.min(minBytes, frame.frameBytes());
            maxBytes = Math.max(maxBytes, frame.frameBytes());
            sumBytes += frame.frameBytes();
            minBoxes = Math.min(minBoxes, frame.boxes().size());
            maxBoxes = Math.max(maxBoxes, frame.boxes().size());
            sumBoxes += frame.boxes().size();
            int tags = baseReference.get(i).size();
            minTags = Math.min(minTags, tags);
            maxTags = Math.max(maxTags, tags);
            sumTags += tags;
        }
        double count = sceneFrames.length;
        line(
                "frame set: %d different frames from %d real pictures (%d frames for each picture),"
                        + " %dx%d %s. Made in %.0f s.",
                sceneFrames.length,
                names.size(),
                framesForEachSource,
                WIDTH,
                HEIGHT,
                GRAY_CAMERA ? "uncompressed 8-bit gray (GRAY8)" : "monochrome MJPEG 4:2:2",
                (System.nanoTime() - start) / 1e9);
        line(
                "frame set: frame size (%s) min/mean/max = %d/%.0f/%d kB | model regions for each"
                        + " frame min/mean/max = %d/%.1f/%d | decoded tags for each frame"
                        + " min/mean/max = %d/%.1f/%d",
                CAMERA_FORMAT,
                minBytes / 1000,
                sumBytes / count / 1000.0,
                maxBytes / 1000,
                minBoxes,
                sumBoxes / count,
                maxBoxes,
                minTags,
                sumTags / count,
                maxTags);
        if (System.getenv("PV_MLTAG_SIM_DETAIL") != null) {
            for (int source = 0; source < names.size(); source++) {
                int frameCount = 0;
                int boxes = 0;
                int tags = 0;
                long bytes = 0;
                for (int i = 0; i < sceneFrames.length; i++) {
                    if (sceneFrames[i].source() == source) {
                        frameCount++;
                        boxes += sceneFrames[i].boxes().size();
                        tags += baseReference.get(i).size();
                        bytes += sceneFrames[i].frameBytes();
                    }
                }
                line(
                        "  source %2d %-44s regions=%.1f (%s) decodedTags=%.1f frameKb=%d",
                        source,
                        names.get(source),
                        boxes / (double) frameCount,
                        fromLabels.get(source) ? "labels" : "detector",
                        tags / (double) frameCount,
                        bytes / frameCount / 1000);
            }
        }
    }

    /** The tag data of a result: for each tag the ID, the 4 corners, and the pose. Sorted. */
    private static List<double[]> tagData(CVPipelineResult result) {
        List<double[]> tags = new ArrayList<>();
        for (TrackedTarget target : result.targets) {
            double[] values = new double[15];
            values[0] = target.getFiducialId();
            var corners = target.getTargetCorners();
            for (int i = 0; i < 4 && i < corners.size(); i++) {
                values[1 + i * 2] = corners.get(i).x;
                values[2 + i * 2] = corners.get(i).y;
            }
            var pose = target.getBestCameraToTarget3d();
            if (pose != null) {
                values[9] = pose.getX();
                values[10] = pose.getY();
                values[11] = pose.getZ();
                values[12] = pose.getRotation().getX();
                values[13] = pose.getRotation().getY();
                values[14] = pose.getRotation().getZ();
            }
            tags.add(values);
        }
        tags.sort((a, b) -> a[0] != b[0] ? Double.compare(a[0], b[0]) : Double.compare(a[1], b[1]));
        return tags;
    }

    /**
     * Compares the tag data of a result with the reference data of the same frame: all 15 values of
     * each tag (ID, corners, pose). A NaN value is always a difference, also when the reference has
     * NaN too: a NaN corner or pose is never a valid result, and a plain comparison with NaN is
     * false, which would count it as equal.
     *
     * @param reference The reference data of the pose iteration count of the run.
     * @return null if the data is equal, or a text that tells the difference.
     */
    static String compareWithReference(
            List<List<double[]>> reference, int index, List<double[]> tags) {
        List<double[]> expected = reference.get(index);
        if (expected == null) {
            return "frame " + index + ": no reference data";
        }
        if (expected.size() != tags.size()) {
            return "frame " + index + ": " + tags.size() + " tags, reference " + expected.size();
        }
        for (int tag = 0; tag < tags.size(); tag++) {
            for (int i = 0; i < 15; i++) {
                double a = expected.get(tag)[i];
                double b = tags.get(tag)[i];
                if (Double.isNaN(a) || Double.isNaN(b) || Math.abs(a - b) > 1e-9) {
                    return "frame "
                            + index
                            + " tag "
                            + (int) expected.get(tag)[0]
                            + " value "
                            + i
                            + ": "
                            + b
                            + ", reference "
                            + a;
                }
            }
        }
        return null;
    }

    /** Makes the reference for an iteration count if it is not there yet. */
    private static synchronized List<List<double[]>> ensureReference(int numIterations)
            throws Exception {
        List<List<double[]>> reference = referenceByIterations.get(numIterations);
        return reference != null ? reference : prepareReference(numIterations);
    }

    /**
     * Makes the reference data for a pose iteration count: a slow camera gives each frame, and a
     * pipeline with one lane and no NPU wait completes each frame before the next frame arrives. The
     * frames go through the same capture library (and in the mjpeg format the same JPEG decoder) as
     * in the load runs. Only the iteration count is different from the base settings: the model
     * gives no duplicate regions and does no extra input preparation, because these do not change
     * the tag data.
     */
    private static synchronized List<List<double[]>> prepareReference(int numIterations)
            throws Exception {
        // The calm run must not have the extra decode work of an earlier scenario.
        setExtraDecodePercent(0);
        List<List<double[]>> reference = new ArrayList<>();
        for (int i = 0; i < sceneFrames.length; i++) {
            reference.add(null);
        }
        var properties = new FrameStaticProperties(WIDTH, HEIGHT, 80.0, makeCalibration());
        final long sleepMicros = 30_000;
        GStreamerCapture capture =
                GStreamerCapture.open(cameraPipeline(sleepMicros, 0), GStreamerCaptureJNI.OUTPUT_GRAY);
        assertNotNull(capture, "Could not open the simulated camera");
        SimulatedCamera camera = new SimulatedCamera(capture, properties);
        AprilTagPipelineSettings settings = makeSettings(1.0);
        settings.numIterations = numIterations;
        AprilTagPipeline pipeline =
                new SimulatedPipeline(
                        settings, 1, new SimulatedModel(new SimulatedNpu(1, 0, 0.0, 0), 1.0));
        int missing = sceneFrames.length;
        long deadline =
                System.nanoTime() + sceneFrames.length * sleepMicros * 1000L * 5 + 30_000_000_000L;
        try {
            while (missing > 0 && System.nanoTime() < deadline) {
                camera.requestFrameThresholdType(pipeline.getThresholdType());
                camera.requestFrameRotation(ImageRotationMode.DEG_0);
                camera.requestFrameCopies(false, false);
                camera.requestMonochromeInput(true);
                Frame frame = camera.get();
                if (frame.colorImage.getMat().empty()) {
                    frame.release();
                    continue;
                }
                CVPipelineResult result = pipeline.run(frame, QuirkyCamera.DefaultCamera);
                if (result == null) {
                    continue;
                }
                int index = readFrameIndex(result.inputAndOutputFrame.colorImage.getMat());
                assertTrue(index >= 0, "A frame has no valid index");
                List<double[]> tags = tagData(result);
                if (reference.get(index) == null) {
                    // A NaN value can never be equal, so the load runs could not pass. Say it here,
                    // where the cause is clear.
                    for (double[] values : tags) {
                        for (double value : values) {
                            assertTrue(
                                    !Double.isNaN(value), "The reference data of frame " + index + " has NaN");
                        }
                    }
                    reference.set(index, tags);
                    missing--;
                } else {
                    // The same frame must always give the same data.
                    String difference = compareWithReference(reference, index, tags);
                    assertTrue(
                            difference == null,
                            "The reference run is not stable ("
                                    + numIterations
                                    + " pose iterations): "
                                    + difference);
                }
                result.release();
            }
        } finally {
            pipeline.release();
            camera.release();
        }
        assertEquals(0, missing, "The reference run did not get each frame");
        assertEquals(0, unknownFrames.get(), "The simulated model could not identify a frame");
        referenceByIterations.put(numIterations, reference);
        return reference;
    }

    // ---------------------------------------------------------------------------------------
    // Simulated camera
    // ---------------------------------------------------------------------------------------

    /**
     * @param sleepMicros Wait between two frames.
     * @param firstFrame Index of the first frame. The camera gives the frames of the set in sequence
     *     and starts again at frame 0 after the last frame.
     */
    private static String cameraPipeline(long sleepMicros, int firstFrame) {
        // identity waits between buffers, which makes a frame rate. The name pvcapturetime tells
        // the capture library that a frame "arrives" after this element.
        // In the gray mode the caps tell GStreamer the raw format, because a raw file has no header.
        // A real camera in this mode gives the same caps through v4l2src.
        String caps =
                GRAY_CAMERA
                        ? "video/x-raw,format=GRAY8,width=" + WIDTH + ",height=" + HEIGHT + ",framerate=60/1"
                        : "image/jpeg,framerate=60/1";
        return "multifilesrc location=\""
                + frameDirectory.resolve(framePattern())
                + "\" index="
                + firstFrame
                + " start-index=0 stop-index="
                + (sceneFrames.length - 1)
                + " loop=true caps="
                + caps
                + " ! identity name=pvcapturetime sleep-time="
                + sleepMicros
                + " ! appsink name=pvappsink emit-signals=false max-buffers=1 drop=true sync=false";
    }

    /** Finds the identity sleep time that gives the requested frame rate on this computer. */
    private static long calibrateCameraSleepMicros(double fps) throws InterruptedException {
        long targetMicros = Math.round(1e6 / fps);
        long sleepMicros = targetMicros;
        for (int pass = 0; pass < 3; pass++) {
            try (GStreamerCapture capture =
                    GStreamerCapture.open(cameraPipeline(sleepMicros, 0), GStreamerCaptureJNI.OUTPUT_GRAY)) {
                assertNotNull(capture);
                Thread.sleep(300);
                long before = capture.getStats().sourceBuffers();
                long start = System.nanoTime();
                Thread.sleep(1200);
                double measuredFps =
                        (capture.getStats().sourceBuffers() - before) / ((System.nanoTime() - start) / 1e9);
                double measuredMicros = 1e6 / measuredFps;
                sleepMicros = Math.max(1, Math.round(sleepMicros - (measuredMicros - targetMicros)));
            }
        }
        return sleepMicros;
    }

    /** The frame provider of the simulation. It is the grab code of GStreamerUSBFrameProvider. */
    private static class SimulatedCamera extends CpuImageProcessor {
        final GStreamerCapture capture;
        final FrameStaticProperties properties;

        SimulatedCamera(GStreamerCapture capture, FrameStaticProperties properties) {
            this.capture = capture;
            this.properties = properties;
        }

        @Override
        CapturedFrame getInputMat() {
            GStreamerFrame frame = capture.grab(250);
            if (frame == null) {
                return new CapturedFrame(new CVMat(), properties, 0);
            }
            return new CapturedFrame(
                    frame.toCVMat(),
                    properties,
                    frame.toCaptureMetadata(FrameCaptureMetadata.SOURCE_GSTREAMER_JNI));
        }

        @Override
        public String getName() {
            return "SimulatedCamera";
        }

        @Override
        public void release() {
            capture.close();
            releaseProcessorResources();
        }

        @Override
        protected boolean checkCameraConnected() {
            return true;
        }
    }

    // ---------------------------------------------------------------------------------------
    // Simulated NPU
    // ---------------------------------------------------------------------------------------

    /**
     * The CPU time that the NPU library uses for one inference (copy of the input, conversion of the
     * model output, box decode, NMS), as milliseconds of this computer. The default of 2.0 ms is an
     * upper limit: the same work for a model with one class (copy of 640x640x3 bytes, conversion of
     * 554,400 output values to float, scan of 8,400 cells, box decode and NMS of 20 to 300
     * candidates) takes 0.5 to 0.7 ms on the computer where this simulation was made, and 2.0 to 2.3
     * ms for a model with 80 classes. PV_MLTAG_SIM_NPU_CPU_MS changes it. The thread is busy for this
     * time and waits for the remaining part of the inference time.
     */
    private static final double NPU_CPU_MILLIS = npuCpuMillis();

    private static double npuCpuMillis() {
        String raw = System.getenv("PV_MLTAG_SIM_NPU_CPU_MS");
        return raw == null || raw.isBlank() ? 2.0 : Double.parseDouble(raw.trim());
    }

    private static class SimulatedNpu {
        final Semaphore cores;
        final long inferenceNanos;
        final double spread;
        final long busyNanos;

        /**
         * @param cores Number of inferences at the same time.
         * @param inferenceNanos Mean time of one inference, with the CPU time of the NPU library.
         * @param spread Standard deviation of the inference time, as a part of the mean.
         * @param busyNanos The part of the inference time where the calling thread uses the CPU.
         */
        SimulatedNpu(int cores, long inferenceNanos, double spread, long busyNanos) {
            this.cores = new Semaphore(cores, true);
            this.inferenceNanos = inferenceNanos;
            this.spread = spread;
            this.busyNanos = Math.min(inferenceNanos, busyNanos);
        }

        /** The time of the next inference: a normal distribution, limited to 0.75 to 1.4 x mean. */
        long nextInferenceNanos() {
            if (spread <= 0) {
                return inferenceNanos;
            }
            double factor = 1.0 + spread * ThreadLocalRandom.current().nextGaussian();
            return Math.round(inferenceNanos * Math.max(0.75, Math.min(1.4, factor)));
        }
    }

    /**
     * A model for the simulated NPU. The detector does the real input preparation of the Rubik and
     * RKNN detectors, waits for an NPU core, waits for the inference time, and gives the regions of
     * the frame that it got (it reads the frame index from the picture).
     */
    private static class SimulatedModel implements Model {
        final SimulatedNpu npu;
        final double workMultiplier;

        /**
         * @param workMultiplier 1 = the real CPU work. A larger value repeats the input preparation and
         *     gives a part of the regions more than one time, so the tag decode does this multiple of
         *     its work. The decode pipe removes the duplicate tags.
         */
        SimulatedModel(SimulatedNpu npu, double workMultiplier) {
            this.npu = npu;
            this.workMultiplier = workMultiplier;
        }

        @Override
        public ObjectDetector load() {
            Model model = this;
            return new ObjectDetector() {
                private final Letterbox.ModelInputScratch scratch = new Letterbox.ModelInputScratch();
                private final Mat rgb = new Mat();
                private volatile long busySink = 0;
                private double extraPrepare = 0;
                private double extraBoxes = 0;
                private int nextExtraBox = 0;

                @Override
                public Model getModel() {
                    return model;
                }

                @Override
                public List<String> getClasses() {
                    return List.of("tag");
                }

                @Override
                public List<NeuralNetworkPipeResult> detect(Mat in, double nmsThresh, double boxThresh) {
                    int frameIndex = readFrameIndex(in);
                    if (frameIndex < 0) {
                        unknownFrames.incrementAndGet();
                    }
                    List<RotatedRect> boxes =
                            frameIndex < 0 ? List.<RotatedRect>of() : sceneFrames[frameIndex].boxes();

                    extraPrepare += workMultiplier - 1.0;
                    int prepareCount = 1;
                    while (extraPrepare >= 1.0) {
                        prepareCount++;
                        extraPrepare -= 1.0;
                    }
                    for (int i = 0; i < prepareCount; i++) {
                        // The same input preparation as RubikObjectDetector and RknnObjectDetector.
                        // PV_MLTAG_MONO_MODEL_INPUT=exact gives the time of a board where the
                        // self-check fails and the 3-channel base operation runs.
                        Letterbox.ModelInput input =
                                Letterbox.prepareModelInput(
                                        in, scratch, new Size(640, 640), ColorHelper.colorToScalar(Color.GRAY));
                        // The JNI libraries convert BGR to RGB.
                        Imgproc.cvtColor(input.image(), rgb, Imgproc.COLOR_BGR2RGB);
                    }
                    modelInputPath =
                            in.channels() != 1
                                    ? "color"
                                    : scratch.lastWasMonochromeFastPath()
                                            ? "monochrome fast path"
                                            : "monochrome base operation";

                    try {
                        npu.cores.acquire();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        return List.of();
                    }
                    try {
                        long begin = System.nanoTime();
                        long deadline = begin + npu.nextInferenceNanos();
                        // The CPU part of the NPU library: the thread is busy.
                        long busyEnd = begin + npu.busyNanos;
                        long spin = 0;
                        while (System.nanoTime() < busyEnd) {
                            spin += 31 * spin + 7;
                        }
                        busySink = spin;
                        long remaining;
                        while ((remaining = deadline - System.nanoTime()) > 0) {
                            LockSupport.parkNanos(remaining);
                        }
                    } finally {
                        npu.cores.release();
                    }

                    List<NeuralNetworkPipeResult> results = new ArrayList<>(boxes.size());
                    for (RotatedRect box : boxes) {
                        results.add(
                                new NeuralNetworkPipeResult(
                                        new RotatedRect(box.center.clone(), box.size.clone(), box.angle), 0, 0.9));
                    }
                    // More tag decode work: give a part of the boxes again.
                    extraBoxes += (workMultiplier - 1.0) * boxes.size();
                    while (extraBoxes >= 1.0 && !boxes.isEmpty()) {
                        RotatedRect box = boxes.get(nextExtraBox++ % boxes.size());
                        results.add(
                                new NeuralNetworkPipeResult(
                                        new RotatedRect(box.center.clone(), box.size.clone(), box.angle), 0, 0.9));
                        extraBoxes -= 1.0;
                    }
                    return results;
                }

                @Override
                public void release() {
                    scratch.release();
                    rgb.release();
                }
            };
        }

        @Override
        public String getUID() {
            return "simulated";
        }

        @Override
        public String getNickname() {
            return "simulated";
        }

        @Override
        public Family getFamily() {
            return Family.RKNN;
        }

        @Override
        public ModelProperties getProperties() {
            return null;
        }
    }

    private static class SimulatedPipeline extends AprilTagPipeline {
        private final Model model;

        SimulatedPipeline(AprilTagPipelineSettings settings, int lanes, Model model) {
            super(settings, lanes);
            this.model = model;
        }

        @Override
        protected boolean checkMLAvailability() {
            return true;
        }

        @Override
        protected Model resolveModel(AprilTagPipelineSettings settings) {
            return model;
        }
    }

    private static AprilTagPipelineSettings makeSettings(double workMultiplier) {
        var settings = new AprilTagPipelineSettings();
        settings.useMLDetection = true;
        // No frame decimation. ATR is a setting of the pipeline (default: on): a tag region that is
        // wider than atrTargetDimension pixels is made smaller before the tag decode.
        // PV_MLTAG_SIM_ATR=0 sets it off: then each tag is decoded at the full camera resolution.
        settings.atrEnabled = ATR_ENABLED;
        settings.decimate = 1;
        settings.threads = 4;
        settings.refineEdges = true;
        settings.tagFamily = AprilTagFamily.kTag36h11;
        settings.targetModel = TargetModel.kAprilTag6p5in_36h11;
        settings.mlRoiPaddingPixels = ROI_PADDING;
        // 3D mode. The default of a new pipeline: one orthogonal-iteration pose for each tag.
        // PV_MLTAG_SIM_MULTITAG=1 adds the multi-tag pose and keeps the single-tag poses.
        settings.solvePNPEnabled = true;
        boolean multiTag = "1".equals(System.getenv("PV_MLTAG_SIM_MULTITAG"));
        settings.doMultiTarget = multiTag;
        settings.doSingleTargetAlways = multiTag;
        // The pose time is almost proportional to the number of iterations.
        settings.numIterations = iterationsFor(workMultiplier);
        settings.inputShouldShow = false;
        settings.outputShouldShow = false;
        settings.model =
                new ModelProperties(
                        Path.of("test", "simulated.rknn").toAbsolutePath(),
                        "simulated",
                        List.of("tag"),
                        640,
                        640,
                        Family.RKNN,
                        Version.YOLOV11);
        return settings;
    }

    /** A pinhole calibration for an 80 degree lens at 1600x1304. */
    private static CameraCalibrationCoefficients makeCalibration() {
        double focal = (WIDTH / 2.0) / Math.tan(Math.toRadians(40.0));
        return new CameraCalibrationCoefficients(
                new Size(WIDTH, HEIGHT),
                new JsonMatOfDouble(
                        3, 3, new double[] {focal, 0, WIDTH / 2.0, 0, focal, HEIGHT / 2.0, 0, 0, 1}),
                new JsonMatOfDouble(1, 8, new double[] {0, 0, 0, 0, 0, 0, 0, 0}),
                new double[0],
                List.of(),
                new Size(0, 0),
                0,
                CameraLensModel.LENSMODEL_OPENCV);
    }

    // ---------------------------------------------------------------------------------------
    // Vision loop
    // ---------------------------------------------------------------------------------------

    /**
     * Checks the capture time, the order and the sequence ID of one result.
     *
     * @param resultCaptureNanos The capture time of the result.
     * @param frameTimestampNanos The time stamp of the frame of the result.
     * @param previousResultCaptureNanos The capture time of the previous result of the same camera,
     *     or Long.MIN_VALUE for the first result.
     * @param resultSequenceID The sequence ID of the result.
     * @param frameSequenceID The sequence ID of the frame of the result.
     * @return null if the result is correct, or a text that tells the error.
     */
    static String orderAndTimeError(
            long resultCaptureNanos,
            long frameTimestampNanos,
            long previousResultCaptureNanos,
            long resultSequenceID,
            long frameSequenceID) {
        // The robot code uses this time with the pose. A result for an older frame with the time of
        // the newest frame gives a pose with a time that is too new.
        if (resultCaptureNanos != frameTimestampNanos) {
            return "capture time "
                    + resultCaptureNanos
                    + " is not the time of the result frame "
                    + frameTimestampNanos;
        }
        // Strictly larger: a result out of capture order or a result given two times fails here.
        if (resultCaptureNanos <= previousResultCaptureNanos) {
            return "capture time "
                    + resultCaptureNanos
                    + " is not after the previous result "
                    + previousResultCaptureNanos;
        }
        if (resultSequenceID != frameSequenceID) {
            return "sequence ID "
                    + resultSequenceID
                    + " is not the ID of the result frame "
                    + frameSequenceID;
        }
        return null;
    }

    /** Makes the pipeline of a camera loop. A test can give a pipeline with a fault. */
    @FunctionalInterface
    private interface PipelineMaker {
        AprilTagPipeline make(AprilTagPipelineSettings settings, int lanes, Model model);
    }

    /** The loop of VisionRunner for one camera, with measurement. */
    private static class CameraLoop implements Runnable {
        final SimulatedCamera camera;
        final AprilTagPipeline pipeline;
        final AtomicBoolean running = new AtomicBoolean(true);
        final AtomicBoolean measuring = new AtomicBoolean(false);

        /** The reference data of the pose iteration count of this pipeline. */
        final List<List<double[]>> reference;

        /** Results in the measurement. The FPS values use it. */
        long results = 0;

        /** Results that were compared with the reference: each result from the first one. */
        long checkedResults = 0;

        /** Compared results that were equal to the reference. */
        long checkedEqual = 0;

        String firstDifference;

        /** Results with a wrong capture time, a wrong order or a wrong sequence ID. */
        long orderAndTimeErrors = 0;

        String firstOrderAndTimeError;
        long previousResultCaptureNanos = Long.MIN_VALUE;
        long previousNativeCaptureNanos = -1;
        long previousNativeSequence = -1;
        long getNanos = 0;
        long runNanos = 0;
        long pipeNanos = 0;
        final List<Long> latencyNanos = new ArrayList<>();
        final List<Long> resultTimesNanos = new ArrayList<>();

        /** For each result: the time that the pipeline call took in the vision thread. */
        final List<Long> resultRunNanos = new ArrayList<>();

        final List<Double> reportedFps = new ArrayList<>();

        /** For each result: the source of its frame. */
        final List<Integer> resultSceneTypes = new ArrayList<>();

        /** For each result: the time of the pipeline for its frame. */
        final List<Long> resultPipeNanos = new ArrayList<>();

        Throwable failure;

        CameraLoop(
                SimulatedCamera camera, AprilTagPipeline pipeline, List<List<double[]>> reference) {
            this.camera = camera;
            this.pipeline = pipeline;
            this.reference = reference;
        }

        @Override
        public void run() {
            try {
                while (running.get()) {
                    camera.requestFrameThresholdType(
                            pipeline != null ? pipeline.getThresholdType() : FrameThresholdType.NONE);
                    camera.requestFrameRotation(ImageRotationMode.DEG_0);
                    camera.requestFrameCopies(false, false);
                    camera.requestMonochromeInput(true);

                    long getStart = System.nanoTime();
                    Frame frame = camera.get();
                    long runStart = System.nanoTime();
                    if (frame.colorImage.getMat().empty()) {
                        frame.release();
                        continue;
                    }

                    if (pipeline == null) {
                        // Reference loop: take each frame and do no work. It shows the frame time
                        // variation of the simulated camera and the JPEG decode alone. There is no
                        // tag data to compare, but the counts must not be 0, or the fraction of
                        // equal results is 0 / 0.
                        long taken = System.nanoTime();
                        checkedResults++;
                        checkedEqual++;
                        if (measuring.get()) {
                            results++;
                            resultSceneTypes.add(
                                    sceneFrames[Math.max(0, readFrameIndex(frame.colorImage.getMat()))].source());
                            resultPipeNanos.add(0L);
                            getNanos += runStart - getStart;
                            latencyNanos.add(
                                    MathUtils.wpiNanoTime() - frame.captureMetadata.captureTimestampNanos());
                            resultTimesNanos.add(taken);
                            resultRunNanos.add(0L);
                        }
                        frame.release();
                        continue;
                    }

                    CVPipelineResult result = pipeline.run(frame, QuirkyCamera.DefaultCamera);
                    long runEnd = System.nanoTime();
                    if (measuring.get()) {
                        getNanos += runStart - getStart;
                        runNanos += runEnd - runStart;
                    }
                    if (result == null) {
                        continue;
                    }
                    long doneNanos = MathUtils.wpiNanoTime();
                    // The tag data must be equal to the data of the calm run of this frame. This is
                    // checked from the first result, not only in the measurement: the pipeline
                    // changes its inference mode in the warm-up, and those results must be correct
                    // too.
                    int index = readFrameIndex(result.inputAndOutputFrame.colorImage.getMat());
                    String difference =
                            index < 0
                                    ? "a result frame has no valid index"
                                    : compareWithReference(reference, index, tagData(result));
                    checkedResults++;
                    if (difference == null) {
                        checkedEqual++;
                    } else if (firstDifference == null) {
                        firstDifference = difference;
                    }
                    String orderError =
                            orderAndTimeError(
                                    result.getImageCaptureTimestampNanos(),
                                    result.inputAndOutputFrame.timestampNanos,
                                    previousResultCaptureNanos,
                                    result.sequenceID,
                                    result.inputAndOutputFrame.sequenceID);
                    // The capture time of the native library (monotonic clock), computed in one
                    // native call from the age of the frame. The error text gives it, so a reader
                    // can see if the frames were out of order or only the Java time stamp moved.
                    FrameCaptureMetadata metadata = result.inputAndOutputFrame.captureMetadata;
                    long nativeCaptureNanos =
                            metadata.estimatedAgeNanos() >= 0
                                    ? metadata.nativePullTimestampNanos() - metadata.estimatedAgeNanos()
                                    : -1;
                    if (orderError != null) {
                        orderAndTimeErrors++;
                        if (firstOrderAndTimeError == null) {
                            firstOrderAndTimeError =
                                    orderError
                                            + " (result "
                                            + checkedResults
                                            + ", frame index "
                                            + index
                                            + ", native capture time "
                                            + nativeCaptureNanos
                                            + " sequence "
                                            + metadata.nativeSequence()
                                            + ", previous result: native capture time "
                                            + previousNativeCaptureNanos
                                            + " sequence "
                                            + previousNativeSequence
                                            + ")";
                        }
                    }
                    previousResultCaptureNanos = result.getImageCaptureTimestampNanos();
                    previousNativeCaptureNanos = nativeCaptureNanos;
                    previousNativeSequence = metadata.nativeSequence();
                    if (measuring.get()) {
                        pipeNanos += (long) result.processingNanos;
                        results++;
                        resultSceneTypes.add(index < 0 ? 0 : sceneFrames[index].source());
                        resultPipeNanos.add((long) result.processingNanos);
                        latencyNanos.add(doneNanos - result.getImageCaptureTimestampNanos());
                        resultTimesNanos.add(runEnd);
                        resultRunNanos.add(runEnd - runStart);
                        reportedFps.add(result.fps);
                    }
                    result.release();
                }
            } catch (Throwable t) {
                failure = t;
            }
        }
    }

    private record Outcome(
            double cameraFps,
            double[] resultFps,
            double latencyP50Millis,
            double latencyP95Millis,
            double latencyMaxMillis,
            double equalToReferenceFraction,
            String firstDifference,
            double decodeMillis,
            double cpuCores,
            double getMillis,
            double runMillis,
            double pipeMillis,
            double[] p99Fps,
            double[] worstSecondFps,
            double[] reportedFpsLow,
            long results,
            long checkedResults,
            long checkedEqual,
            long orderAndTimeErrors,
            String firstOrderAndTimeError,
            int rtThreads,
            int poseIterations) {}

    /** The policy field of a line of /proc/.../stat (0 = SCHED_OTHER, 1 = SCHED_FIFO), or -1. */
    static int schedulingPolicy(String stat) {
        // The thread name (field 2) is in parentheses and can contain spaces, so the fields are
        // counted after the last ')'. The first field there is field 3, so field 41 is index 38.
        String[] fields = stat.substring(stat.lastIndexOf(')') + 1).trim().split(" +");
        return fields.length > 38 ? Integer.parseInt(fields[38]) : -1;
    }

    /** The number of threads of this process with the policy SCHED_FIFO. */
    private static int countRealtimeThreads() throws java.io.IOException {
        int count = 0;
        List<Path> tasks = new ArrayList<>();
        try (var stream = java.nio.file.Files.list(Path.of("/proc/self/task"))) {
            stream.forEach(tasks::add);
        }
        for (Path task : tasks) {
            try {
                if (schedulingPolicy(java.nio.file.Files.readString(task.resolve("stat"))) == 1) {
                    count++;
                }
            } catch (java.io.IOException e) {
                // The thread ended after the list was read.
            }
        }
        return count;
    }

    /** PV_GSTREAMER_CAPTURE_RT_PRIORITY as the native library reads it: 0 to 50, 0 if not set. */
    private static int realtimePrioritySetting() {
        String raw = System.getenv("PV_GSTREAMER_CAPTURE_RT_PRIORITY");
        if (raw == null || raw.isBlank()) {
            return 0;
        }
        try {
            return Math.max(0, Math.min(50, Integer.parseInt(raw.trim())));
        } catch (NumberFormatException e) {
            return 0;
        }
    }

    /**
     * Runs one scenario.
     *
     * @param board The coprocessor profile.
     * @param cameras Number of cameras.
     * @param slowdown The CPU of the coprocessor is this many times slower than this computer.
     * @param sleepMicros Camera pacing from {@link #calibrateCameraSleepMicros(double)}.
     * @param extraDecodePercent The extra JPEG decode work, from {@link #extraDecodePercent(double)}.
     * @param pipelineMaker Makes the pipeline of each camera.
     */
    private static Outcome runScenario(
            Board board,
            int cameras,
            double slowdown,
            long sleepMicros,
            int extraDecodePercent,
            PipelineMaker pipelineMaker)
            throws Exception {
        // Time scaling: all times of this computer are 1 / slowdown of the coprocessor times.
        // Work mode: real time, and the CPU stages do "slowdown" times their work. The JPEG decode
        // uses its own factor (PV_MLTAG_SIM_JPEG_FACTOR), and the gray mode has no JPEG decode.
        final double timeScale = WORK_MODE ? 1.0 : slowdown;
        final double workMultiplier = WORK_MODE ? slowdown : 1.0;
        final int poseIterations = iterationsFor(workMultiplier);
        // The reference must have the pose iteration count of the load run, so that the pose can be
        // compared. runBoard makes it before the scenario, with no extra decode work.
        List<List<double[]>> reference = referenceByIterations.get(poseIterations);
        assertNotNull(reference, "No reference for " + poseIterations + " pose iterations");
        // The CPU part of the inference is CPU work: it does not get shorter with the time scale,
        // and it is multiplied in the work mode.
        SimulatedNpu npu =
                new SimulatedNpu(
                        board.npuCores(),
                        Math.round(board.inferenceMillis() * 1e6 / timeScale),
                        board.inferenceSpread(),
                        Math.round(NPU_CPU_MILLIS * 1e6 * workMultiplier));

        var properties = new FrameStaticProperties(WIDTH, HEIGHT, 80.0, makeCalibration());
        List<CameraLoop> loops = new ArrayList<>();
        List<Thread> threads = new ArrayList<>();
        List<GStreamerCapture> captures = new ArrayList<>();
        List<AprilTagPipeline> pipelines = new ArrayList<>();
        lastScenarioCaptures = captures;
        // The extra decode work is one setting for the whole process. It is set here, and the
        // finally block below sets it back to 0, also when an assertion or an exception stops the
        // scenario. Without this, later tests in the same JVM would decode each frame more times.
        setExtraDecodePercent(extraDecodePercent);
        Throwable primary = null;
        try {
            for (int i = 0; i < cameras; i++) {
                // Each camera starts at a different frame of the set, so the cameras do not have
                // the same view at the same time.
                int firstFrame = (int) ((long) i * sceneFrames.length / cameras);
                GStreamerCapture capture =
                        GStreamerCapture.open(
                                cameraPipeline(sleepMicros, firstFrame), GStreamerCaptureJNI.OUTPUT_GRAY);
                assertNotNull(capture, "Could not open the simulated camera");
                captures.add(capture);
                AprilTagPipeline pipeline =
                        NULL_PIPELINE
                                ? null
                                : pipelineMaker.make(
                                        makeSettings(workMultiplier),
                                        board.lanes(),
                                        new SimulatedModel(npu, workMultiplier));
                pipelines.add(pipeline);
                var loop = new CameraLoop(new SimulatedCamera(capture, properties), pipeline, reference);
                loops.add(loop);
                Thread thread = new Thread(loop, "SimVision-" + i);
                threads.add(thread);
                thread.start();
            }

            var os =
                    (com.sun.management.OperatingSystemMXBean)
                            ManagementFactory.getOperatingSystemMXBean();

            // Warm-up: JIT compile, detector creation, start of the inference overlap. The tag data
            // of the results in the warm-up is compared too (see CameraLoop), only the timing
            // values are not measured.
            Thread.sleep(5000);

            // The capture threads must have the real-time policy that the run scripts give them.
            // The native library only prints one line if the kernel refuses it, which is easy to
            // miss, and then the frame times are not the times of the board.
            int rtThreads = countRealtimeThreads();
            // Each camera has a camera source thread, and in the mjpeg format a JPEG decode thread.
            int expectedRtThreads = cameras * (GRAY_CAMERA ? 1 : 2);
            if (realtimePrioritySetting() > 0 && rtThreads < expectedRtThreads) {
                throw new AssertionError(
                        "The kernel refused SCHED_FIFO (needs root or CAP_SYS_NICE): "
                                + rtThreads
                                + " real-time threads, expected "
                                + expectedRtThreads
                                + ". Remove PV_GSTREAMER_CAPTURE_RT_PRIORITY or give the permission.");
            }

            long[] sourceBefore = new long[cameras];
            long[] producedBefore = new long[cameras];
            long[] convertBefore = new long[cameras];
            for (int i = 0; i < cameras; i++) {
                var stats = captures.get(i).getStats();
                sourceBefore[i] = stats.sourceBuffers();
                producedBefore[i] = stats.framesProduced();
                convertBefore[i] = stats.convertNanosSum();
            }
            long cpuBefore = os.getProcessCpuTime();
            long gcCountBefore = 0;
            long gcMillisBefore = 0;
            for (var gc : ManagementFactory.getGarbageCollectorMXBeans()) {
                gcCountBefore += Math.max(0, gc.getCollectionCount());
                gcMillisBefore += Math.max(0, gc.getCollectionTime());
            }
            long start = System.nanoTime();
            for (CameraLoop loop : loops) {
                loop.measuring.set(true);
            }
            Thread.sleep(measureMillis());
            for (CameraLoop loop : loops) {
                loop.measuring.set(false);
            }
            double seconds = (System.nanoTime() - start) / 1e9;
            double cpuCores = (os.getProcessCpuTime() - cpuBefore) / 1e9 / seconds;
            long gcCount = -gcCountBefore;
            long gcMillis = -gcMillisBefore;
            for (var gc : ManagementFactory.getGarbageCollectorMXBeans()) {
                gcCount += Math.max(0, gc.getCollectionCount());
                gcMillis += Math.max(0, gc.getCollectionTime());
            }

            double cameraFps = 0;
            double decodeMillis = 0;
            for (int i = 0; i < cameras; i++) {
                var stats = captures.get(i).getStats();
                cameraFps += (stats.sourceBuffers() - sourceBefore[i]) / seconds / cameras;
                long produced = stats.framesProduced() - producedBefore[i];
                if (produced > 0) {
                    decodeMillis += (stats.convertNanosSum() - convertBefore[i]) / 1e6 / produced / cameras;
                }
            }

            // Stop the loops before their counters are read. The finally block releases the
            // pipelines and the cameras.
            for (CameraLoop loop : loops) {
                loop.running.set(false);
            }
            for (Thread thread : threads) {
                thread.join(5000);
            }
            for (int i = 0; i < cameras; i++) {
                if (threads.get(i).isAlive()) {
                    throw new AssertionError("The vision loop " + i + " did not stop");
                }
                if (loops.get(i).failure != null) {
                    throw new AssertionError("The vision loop failed", loops.get(i).failure);
                }
            }

            double[] resultFps = new double[cameras];
            double[] p99Fps = new double[cameras];
            double[] worstSecondFps = new double[cameras];
            double[] reportedFpsLow = new double[cameras];
            List<Long> allLatency = new ArrayList<>();
            long results = 0;
            long checkedResults = 0;
            long checkedEqual = 0;
            String firstDifference = null;
            long orderAndTimeErrors = 0;
            String firstOrderAndTimeError = null;
            long getNanos = 0;
            long runNanos = 0;
            long pipeNanos = 0;
            for (int i = 0; i < cameras; i++) {
                CameraLoop loop = loops.get(i);
                getNanos += loop.getNanos;
                runNanos += loop.runNanos;
                pipeNanos += loop.pipeNanos;
                resultFps[i] = loop.results / seconds;
                p99Fps[i] = percentileFps(loop.resultTimesNanos, 0.99, timeScale);
                worstSecondFps[i] = worstWindowFps(loop.resultTimesNanos, timeScale);
                // The FPS value of PhotonVision (CalculateFPSPipe, moving average of 20 frames): the
                // value that 99% of the results are better than.
                double[] reported =
                        loop.reportedFps.stream().mapToDouble(Double::doubleValue).sorted().toArray();
                reportedFpsLow[i] =
                        reported.length > 0 ? reported[(int) (reported.length * 0.01)] / timeScale : 0;
                allLatency.addAll(loop.latencyNanos);
                results += loop.results;
                checkedResults += loop.checkedResults;
                checkedEqual += loop.checkedEqual;
                if (firstDifference == null) {
                    firstDifference = loop.firstDifference;
                }
                orderAndTimeErrors += loop.orderAndTimeErrors;
                if (firstOrderAndTimeError == null) {
                    firstOrderAndTimeError = loop.firstOrderAndTimeError;
                }
            }
            if (System.getenv("PV_MLTAG_SIM_DETAIL") != null) {
                // Times between results, in coprocessor milliseconds: how many are in each range.
                int[] limits = {18, 20, 22, 25, 30, 40, 1000};
                int[] counts = new int[limits.length];
                int total = 0;
                // Long times between results (25 ms or more), and how many of them have a pipeline
                // call that waited 3 ms or more for the result (the result was not ready). The
                // other long times come from a late camera frame.
                int longIntervals = 0;
                int longBecauseOfWait = 0;
                for (CameraLoop loop : loops) {
                    for (int i = 1; i < loop.resultTimesNanos.size(); i++) {
                        double ms =
                                (loop.resultTimesNanos.get(i) - loop.resultTimesNanos.get(i - 1))
                                        / 1e6
                                        * timeScale;
                        if (ms >= 25) {
                            longIntervals++;
                            if (loop.resultRunNanos.get(i) / 1e6 * timeScale >= 3) {
                                longBecauseOfWait++;
                            }
                        }
                        for (int b = 0; b < limits.length; b++) {
                            if (ms < limits[b]) {
                                counts[b]++;
                                break;
                            }
                        }
                        total++;
                    }
                }
                StringBuilder histogram = new StringBuilder();
                for (int b = 0; b < limits.length; b++) {
                    histogram.append(
                            String.format(Locale.ROOT, "<%dms:%.2f%% ", limits[b], 100.0 * counts[b] / total));
                }
                line(
                        "  intervals %s| long: %d, result not ready: %d | gc: %d collections, %d ms",
                        histogram, longIntervals, longBecauseOfWait, gcCount, gcMillis);
                // For each source: the time of the pipeline and the long times between results.
                for (int type = 0; type < sourceNames.size(); type++) {
                    long count = 0;
                    long pipe = 0;
                    long longTimes = 0;
                    for (CameraLoop loop : loops) {
                        for (int i = 1; i < loop.resultTimesNanos.size(); i++) {
                            if (loop.resultSceneTypes.get(i) != type) {
                                continue;
                            }
                            count++;
                            pipe += loop.resultPipeNanos.get(i);
                            double ms =
                                    (loop.resultTimesNanos.get(i) - loop.resultTimesNanos.get(i - 1))
                                            / 1e6
                                            * timeScale;
                            if (ms >= 25) {
                                longTimes++;
                            }
                        }
                    }
                    if (count > 0) {
                        line(
                                "    source %2d %-44s results=%d pipesMs=%.1f longTimes=%.2f%%",
                                type,
                                sourceNames.get(type),
                                count,
                                pipe / 1e6 / count * timeScale,
                                100.0 * longTimes / count);
                    }
                }
            }

            long[] sorted = allLatency.stream().mapToLong(Long::longValue).sorted().toArray();
            assertTrue(sorted.length > 0, "The simulation gave no result");

            // Convert from the time of this computer to the time of the coprocessor.
            for (int i = 0; i < cameras; i++) {
                resultFps[i] /= timeScale;
            }
            return new Outcome(
                    cameraFps / timeScale,
                    resultFps,
                    sorted[sorted.length / 2] / 1e6 * timeScale,
                    sorted[(int) (sorted.length * 0.95)] / 1e6 * timeScale,
                    sorted[sorted.length - 1] / 1e6 * timeScale,
                    (double) checkedEqual / checkedResults,
                    firstDifference,
                    decodeMillis * timeScale,
                    cpuCores,
                    getNanos / 1e6 / results * timeScale,
                    runNanos / 1e6 / results * timeScale,
                    pipeNanos / 1e6 / results * timeScale,
                    p99Fps,
                    worstSecondFps,
                    reportedFpsLow,
                    results,
                    checkedResults,
                    checkedEqual,
                    orderAndTimeErrors,
                    firstOrderAndTimeError,
                    rtThreads,
                    poseIterations);
        } catch (Throwable t) {
            primary = t;
            throw t;
        } finally {
            restoreAfterScenario(loops, threads, captures, pipelines, primary);
        }
    }

    /**
     * Puts the test state back after a scenario: no extra decode work, the loops stopped, and the
     * pipelines and captures released. A pipeline or a capture of a loop that did not stop is not
     * released, because its thread can still use it. That is an error.
     *
     * @param primary The exception that stops the scenario, or null. An error of the cleanup is
     *     added to it, so that the first cause stays visible.
     */
    private static void restoreAfterScenario(
            List<CameraLoop> loops,
            List<Thread> threads,
            List<GStreamerCapture> captures,
            List<AprilTagPipeline> pipelines,
            Throwable primary)
            throws InterruptedException {
        setExtraDecodePercent(0);
        for (CameraLoop loop : loops) {
            loop.running.set(false);
        }
        for (Thread thread : threads) {
            thread.join(5000);
        }
        AssertionError notStopped = null;
        for (int i = 0; i < captures.size(); i++) {
            if (i < threads.size() && threads.get(i).isAlive()) {
                if (notStopped == null) {
                    notStopped =
                            new AssertionError(
                                    "The vision loop "
                                            + i
                                            + " did not stop. Its pipeline and capture are not released.");
                }
                continue;
            }
            AprilTagPipeline pipeline = i < pipelines.size() ? pipelines.get(i) : null;
            if (pipeline != null) {
                pipeline.release();
            }
            if (i < loops.size()) {
                // The simulated camera closes its capture.
                loops.get(i).camera.release();
            } else {
                captures.get(i).close();
            }
        }
        if (notStopped != null) {
            if (primary != null) {
                primary.addSuppressed(notStopped);
            } else {
                throw notStopped;
            }
        }
    }

    /**
     * The frame rate that the given part of the frames is better than. For 0.99: 99% of the times
     * between two results are shorter than 1 / (this rate). Times are converted to the time of the
     * coprocessor.
     */
    private static double percentileFps(
            List<Long> resultTimesNanos, double fraction, double slowdown) {
        if (resultTimesNanos.size() < 3) {
            return 0;
        }
        long[] intervals = new long[resultTimesNanos.size() - 1];
        for (int i = 1; i < resultTimesNanos.size(); i++) {
            intervals[i - 1] = resultTimesNanos.get(i) - resultTimesNanos.get(i - 1);
        }
        Arrays.sort(intervals);
        long interval = intervals[Math.min(intervals.length - 1, (int) (intervals.length * fraction))];
        return 1e9 / (interval * slowdown);
    }

    /** The lowest frame rate in a window of one second of coprocessor time. */
    private static double worstWindowFps(List<Long> resultTimesNanos, double slowdown) {
        if (resultTimesNanos.size() < 3) {
            return 0;
        }
        long window = (long) (1e9 / slowdown);
        long first = resultTimesNanos.get(0);
        long last = resultTimesNanos.get(resultTimesNanos.size() - 1);
        double worst = Double.MAX_VALUE;
        int start = 0;
        int end = 0;
        // Slide the window in steps of one tenth of the window.
        for (long t = first; t + window <= last; t += window / 10) {
            while (start < resultTimesNanos.size() && resultTimesNanos.get(start) < t) {
                start++;
            }
            while (end < resultTimesNanos.size() && resultTimesNanos.get(end) < t + window) {
                end++;
            }
            worst = Math.min(worst, end - start);
        }
        return worst == Double.MAX_VALUE ? 0 : worst;
    }

    /**
     * The number of parallel groups that VisionWorkerPool.maxGroups gives a pipeline of this
     * simulation (settings.threads = 4) with the default PV_MLTAG_ROI_POOL_ADAPTIVE=true. It is the
     * formula of VisionWorkerPool.maxGroups. The report does not call maxGroups, because a call
     * registers the calling thread as one more camera for one second, which changes the group count
     * of the pipelines.
     */
    static int expectedGroups(int poolThreads, int cpus, int cameras) {
        int limit = Math.min(VisionWorkerPool.MAX_GROUPS, poolThreads + 1);
        limit = Math.min(limit, Math.max(1, (cpus - cameras) / cameras));
        return Math.max(1, Math.min(4, limit));
    }

    private static void report(Board board, int cameras, double slowdown, Outcome o) {
        StringBuilder fps = new StringBuilder();
        for (double value : o.resultFps()) {
            fps.append(String.format(Locale.ROOT, "%.1f ", value));
        }
        StringBuilder p99 = new StringBuilder();
        for (double value : o.p99Fps()) {
            p99.append(String.format(Locale.ROOT, "%.1f ", value));
        }
        StringBuilder worstSecond = new StringBuilder();
        for (double value : o.worstSecondFps()) {
            worstSecond.append(String.format(Locale.ROOT, "%.0f ", value));
        }
        StringBuilder reportedLow = new StringBuilder();
        for (double value : o.reportedFpsLow()) {
            reportedLow.append(String.format(Locale.ROOT, "%.1f ", value));
        }
        // The JPEG factor has an effect only in the work mode with the mjpeg format.
        String jpeg =
                WORK_MODE && !GRAY_CAMERA
                        ? " jpegFactor="
                                + java.math.BigDecimal.valueOf(jpegFactor(slowdown))
                                        .stripTrailingZeros()
                                        .toPlainString()
                        : "";
        String notValid = cpuCountNotValid ? " | NOT VALID FOR THE BOARD: " + hostCpus + " CPUs" : "";
        line(
                "%-11s cameras=%d cpuSlowdown=%.1f%s npuMs=%.1f | cameraFps=%.1f resultFps=%s"
                        + "p99FrameFps=%sp99ReportedFps=%sworstSecondFps=%s| latencyMs p50=%.1f p95=%.1f max=%.1f | decodeMs=%.1f getMs=%.1f runMs=%.1f"
                        + " pipesMs=%.1f | equalToReference=%.2f%% (%d results from the first one)"
                        + " orderAndTimeErrors=%d | poseIterations=%d, pose compared | groups=%d"
                        + " rtThreads=%d | cpuCoresBusy=%.2f | modelInput=%s%s",
                board.name(),
                cameras,
                slowdown,
                jpeg,
                board.inferenceMillis(),
                o.cameraFps(),
                fps.toString(),
                p99.toString(),
                reportedLow.toString(),
                worstSecond.toString(),
                o.latencyP50Millis(),
                o.latencyP95Millis(),
                o.latencyMaxMillis(),
                o.decodeMillis(),
                o.getMillis(),
                o.runMillis(),
                o.pipeMillis(),
                o.equalToReferenceFraction() * 100.0,
                o.checkedResults(),
                o.orderAndTimeErrors(),
                o.poseIterations(),
                expectedGroups(VisionWorkerPool.THREADS, appCpus, cameras),
                o.rtThreads(),
                o.cpuCores(),
                modelInputPath,
                notValid);
    }

    private static double[] slowdowns() {
        String raw = System.getenv("PV_MLTAG_SIM_SLOWDOWNS");
        if (raw == null || raw.isBlank()) {
            return new double[] {1.3, 1.5, 2.0};
        }
        return Arrays.stream(raw.split(",")).mapToDouble(Double::parseDouble).toArray();
    }

    private static long measureMillis() {
        String raw = System.getenv("PV_MLTAG_SIM_SECONDS");
        return raw == null || raw.isBlank() ? 6000 : Math.round(Double.parseDouble(raw) * 1000);
    }

    private static int[] intList(String name, int[] defaults) {
        String raw = System.getenv(name);
        if (raw == null || raw.isBlank()) {
            return defaults;
        }
        return Arrays.stream(raw.split(",")).mapToInt(Integer::parseInt).toArray();
    }

    /**
     * Checks the CPU count of the JVM against the board, and applies PV_MLTAG_SIM_APP_CPUS. Call it
     * before prepareScene: then the reference run, the pool threads of VisionWorkerPool and all
     * calibrations run under the CPU mask, as with "taskset java".
     */
    private static synchronized void prepareCpus(Board board) {
        // One time only: the mask stays on the JUnit thread, so a second test in the same JVM would
        // read the masked count.
        if (hostCpus < 0) {
            // On Linux this is the CPU count of the mask of the calling thread.
            hostCpus = Runtime.getRuntime().availableProcessors();
            appCpus = hostCpus;
            // PV_MLTAG_SIM_APP_CPUS=2,3 does what "taskset -c 2,3 java" does for the application:
            // the threads that start from here get these CPUs. Use it with
            // PV_GSTREAMER_CAPTURE_DECODE_CPUS, which gives each decode thread its own CPU. The GC
            // and JIT threads of the JVM keep all CPUs.
            String appCpuSetting = System.getenv("PV_MLTAG_SIM_APP_CPUS");
            if (appCpuSetting != null && !appCpuSetting.isBlank()) {
                long mask = 0;
                for (int cpu : intList("PV_MLTAG_SIM_APP_CPUS", new int[0])) {
                    assertTrue(cpu >= 0 && cpu <= 63, "PV_MLTAG_SIM_APP_CPUS: CPU " + cpu + " is not 0 to 63");
                    mask |= 1L << cpu;
                }
                assertTrue(GStreamerCaptureJNI.setCurrentThreadCpuMask(mask), "Could not set the CPU mask");
                appCpus = Long.bitCount(mask);
            }
        }
        if (hostCpus != board.cpus()) {
            if ("1".equals(System.getenv("PV_MLTAG_SIM_ALLOW_CPU_COUNT"))) {
                cpuCountNotValid = true;
                line(
                        "The JVM can use %d CPUs, the board has %d. The results are NOT VALID for the"
                                + " board.",
                        hostCpus,
                        board.cpus());
            } else {
                throw new AssertionError(
                        "The simulation models a board with "
                                + board.cpus()
                                + " CPUs, but this JVM can use "
                                + hostCpus
                                + ". The number of parallel groups of VisionWorkerPool and the competition"
                                + " for the CPUs are then different from the board. Run: ./gradlew --stop;"
                                + " then taskset -c <"
                                + board.cpus()
                                + " CPUs on "
                                + board.cpus()
                                + " different physical cores, see lscpu -e> ./gradlew --no-daemon"
                                + " :photon-core:test --tests ... A Gradle daemon that already runs keeps its"
                                + " own CPU mask. Set PV_MLTAG_SIM_ALLOW_CPU_COUNT=1 to run anyway.");
            }
        }
    }

    private static String setting(String name) {
        String value = System.getenv(name);
        return value == null || value.isBlank() ? "(not set)" : value;
    }

    /** The checks and the preparation that each simulation test needs before its scenarios. */
    private static void prepareSimulation(Board board) throws Exception {
        assumeTrue(enabled(), "Set PV_MLTAG_SIM=1 to run the simulation");
        assumeTrue(GStreamerCapture.isAvailable(), "Native GStreamer capture library is not loaded");
        assertTrue(
                "mjpeg".equals(CAMERA_FORMAT) || GRAY_CAMERA,
                "PV_MLTAG_SIM_CAMERA_FORMAT must be mjpeg or gray, not " + CAMERA_FORMAT);
        // The gray mode does not use the JPEG decoder of the library.
        if (!GRAY_CAMERA) {
            assumeTrue(GStreamerCaptureJNI.hasCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE));
        }
        prepareCpus(board);
        prepareScene();
    }

    private static void runBoard(Board defaultBoard) throws Exception {
        prepareSimulation(defaultBoard);
        line(
                "mode: %s. Camera format: %s. ATR (adaptive tag resizing): %s. CPU time of the NPU"
                        + " library: %.1f ms for each inference.",
                WORK_MODE ? "real time with more CPU work" : "time scaling",
                GRAY_CAMERA ? "gray (uncompressed GRAY8, no JPEG decode)" : "mjpeg",
                ATR_ENABLED ? "on (pipeline default)" : "off",
                NPU_CPU_MILLIS);
        // The settings that change the result. A run with other settings than the run scripts is
        // then visible in the output.
        line(
                "settings: hostCpus=%d appCpus=%d boardCpus=%d VisionWorkerPool.THREADS=%d"
                        + " PV_OPENCV_THREADS=%s (OpenCV threads %d) PV_GSTREAMER_CAPTURE_RT_PRIORITY=%s"
                        + " PV_GSTREAMER_CAPTURE_JPEG_IDCT=%s PV_GSTREAMER_CAPTURE_DECODE_CPUS=%s"
                        + " PV_MLTAG_SIM_APP_CPUS=%s cameraFormat=%s",
                hostCpus,
                appCpus,
                defaultBoard.cpus(),
                VisionWorkerPool.THREADS,
                setting("PV_OPENCV_THREADS"),
                Core.getNumThreads(),
                setting("PV_GSTREAMER_CAPTURE_RT_PRIORITY"),
                setting("PV_GSTREAMER_CAPTURE_JPEG_IDCT"),
                setting("PV_GSTREAMER_CAPTURE_DECODE_CPUS"),
                setting("PV_MLTAG_SIM_APP_CPUS"),
                CAMERA_FORMAT);

        // PV_MLTAG_SIM_INFERENCE_MS is a list of mean inference times that replace the time of the
        // board (to find the limit of the NPU).
        String inferenceSetting = System.getenv("PV_MLTAG_SIM_INFERENCE_MS");
        double[] inferenceTimes =
                inferenceSetting == null || inferenceSetting.isBlank()
                        ? new double[] {defaultBoard.inferenceMillis()}
                        : Arrays.stream(inferenceSetting.split(","))
                                .mapToDouble(value -> Double.parseDouble(value.trim()))
                                .toArray();

        for (double slowdown : slowdowns()) {
            // The load run of the work mode uses more pose iterations. A calm run with the same
            // count gives its reference, so the pose is compared too. It runs before the camera
            // calibration and with no extra decode work.
            ensureReference(iterationsFor(WORK_MODE ? slowdown : 1.0));
            long sleepMicros = calibrateCameraSleepMicros(WORK_MODE ? CAMERA_FPS : CAMERA_FPS * slowdown);
            for (double inferenceMillis : inferenceTimes) {
                Board board =
                        new Board(
                                defaultBoard.name(),
                                defaultBoard.npuCores(),
                                inferenceMillis,
                                defaultBoard.inferenceSpread(),
                                defaultBoard.lanes(),
                                defaultBoard.cpus());
                for (int cameras : intList("PV_MLTAG_SIM_CAMERAS", new int[] {1, 2})) {
                    Outcome outcome =
                            runScenario(
                                    board,
                                    cameras,
                                    slowdown,
                                    sleepMicros,
                                    extraDecodePercent(slowdown),
                                    SimulatedPipeline::new);
                    report(board, cameras, slowdown, outcome);
                    // The tag data of each result must be equal to the data of the calm run of the
                    // same frame, also at full load and in the warm-up.
                    assertEquals(0, unknownFrames.get(), "The simulated model could not identify a frame");
                    assertEquals(
                            outcome.checkedResults(),
                            outcome.checkedEqual(),
                            "A result is not equal to the reference: " + outcome.firstDifference());
                    // The comparison must include the results of the warm-up, so it must have more
                    // results than the measurement.
                    if (!NULL_PIPELINE) {
                        assertTrue(
                                outcome.checkedResults() > outcome.results(),
                                "The warm-up results were not compared");
                    }
                    assertEquals(
                            0,
                            outcome.orderAndTimeErrors(),
                            "A result has a wrong capture time, order or sequence ID: "
                                    + outcome.firstOrderAndTimeError());
                }
            }
        }
    }

    @Test
    public void rubikPi3() throws Exception {
        runBoard(RUBIK);
    }

    @Test
    public void orangePi5() throws Exception {
        runBoard(ORANGE_PI);
    }

    /** One frame with one tag: ID, 4 corners, pose (15 values). */
    private static List<List<double[]>> oneTagReference() {
        double[] values = new double[15];
        for (int i = 0; i < values.length; i++) {
            values[i] = 1.0 + i * 0.25;
        }
        List<List<double[]>> reference = new ArrayList<>();
        reference.add(List.of(values));
        return reference;
    }

    private static List<double[]> changed(List<List<double[]>> reference, int value, double newValue) {
        double[] copy = reference.get(0).get(0).clone();
        copy[value] = newValue;
        return List.of(copy);
    }

    @Test
    public void compareWithReferenceComparesThePose() {
        List<List<double[]>> reference = oneTagReference();
        double pose = reference.get(0).get(0)[12];
        double corner = reference.get(0).get(0)[3];
        assertNull(compareWithReference(reference, 0, changed(reference, 12, pose)));
        // A change of a pose value must be found in each mode: there is no path that skips it.
        String poseDifference = compareWithReference(reference, 0, changed(reference, 12, pose + 1e-6));
        assertNotNull(poseDifference);
        assertTrue(poseDifference.contains("value 12"), poseDifference);
        assertNotNull(compareWithReference(reference, 0, changed(reference, 3, corner + 1e-6)));
    }

    @Test
    public void compareWithReferenceDoesNotAcceptNaN() {
        List<List<double[]>> reference = oneTagReference();
        // A NaN corner or pose in the result.
        assertNotNull(compareWithReference(reference, 0, changed(reference, 5, Double.NaN)));
        assertNotNull(compareWithReference(reference, 0, changed(reference, 12, Double.NaN)));
        // NaN in the reference and a number in the result.
        List<List<double[]>> nanReference = new ArrayList<>();
        nanReference.add(changed(reference, 12, Double.NaN));
        assertNotNull(compareWithReference(nanReference, 0, reference.get(0)));
        // NaN in both: a NaN pose is never a valid equal result.
        assertNotNull(compareWithReference(nanReference, 0, changed(reference, 12, Double.NaN)));
    }

    @Test
    public void orderAndTimeErrorFindsWrongTimeOrderAndDuplicate() {
        // A correct sequence: each result has the time and the ID of its own frame, in order.
        assertNull(orderAndTimeError(100, 100, Long.MIN_VALUE, 1, 1));
        assertNull(orderAndTimeError(117, 117, 100, 2, 2));
        // The time of the newest input frame on the result of an older frame.
        assertNotNull(orderAndTimeError(134, 117, 100, 2, 2));
        // Two results in the wrong order.
        assertNotNull(orderAndTimeError(100, 100, 117, 1, 1));
        // The same result two times.
        assertNotNull(orderAndTimeError(117, 117, 117, 2, 2));
        // The sequence ID of a different frame.
        assertNotNull(orderAndTimeError(117, 117, 100, 3, 2));
    }

    @Test
    public void schedulingPolicyAndGroupsAreRead() {
        // A line of /proc/<pid>/task/<tid>/stat with a thread name that has a space and a ')'.
        // Field 40 (rt_priority) is 10 and field 41 (policy) is 1 = SCHED_FIFO.
        StringBuilder stat = new StringBuilder("1234 (pv jpeg) decode) S");
        for (int field = 4; field <= 52; field++) {
            stat.append(' ').append(field == 40 ? 10 : field == 41 ? 1 : 0);
        }
        assertEquals(1, schedulingPolicy(stat.toString()));
        assertEquals(0, schedulingPolicy(stat.toString().replace(" 10 1 ", " 0 0 ")));
        // The formula of VisionWorkerPool.maxGroups for 4 CPUs (3 pool threads), as on the board.
        assertEquals(3, expectedGroups(3, 4, 1));
        assertEquals(1, expectedGroups(3, 4, 2));
        // 8 CPUs give more groups: this is why the CPU count of the JVM must be the board count.
        assertEquals(3, expectedGroups(7, 8, 2));
        assertEquals(4, expectedGroups(7, 8, 1));
    }

    /**
     * A pipeline with the fault of the base code: the result gets the time of the input frame of the
     * call, not the time of the frame that the result is for.
     */
    private static class InputFrameTimePipeline extends SimulatedPipeline {
        InputFrameTimePipeline(AprilTagPipelineSettings settings, int lanes, Model model) {
            super(settings, lanes, model);
        }

        @Override
        public CVPipelineResult run(Frame frame, QuirkyCamera cameraQuirks) {
            CVPipelineResult result = super.run(frame, cameraQuirks);
            if (result != null) {
                result.setImageCaptureTimestampNanos(frame.timestampNanos);
            }
            return result;
        }
    }

    /**
     * Proof that the capture time check works: with the input frame time on each result, the
     * results of the overlap mode (a result for an older frame) must give errors. Needs
     * PV_MLTAG_SIM=1.
     */
    @Test
    public void inputFrameTimeOnTheResultIsFound() throws Exception {
        prepareSimulation(ORANGE_PI);
        assumeTrue(!NULL_PIPELINE, "The null pipeline gives no results");
        ensureReference(iterationsFor(1.0));
        long sleepMicros = calibrateCameraSleepMicros(CAMERA_FPS);
        // k = 1 in both modes: the inference time of the Orange Pi is longer than the camera period,
        // so the pipeline uses the overlap mode.
        Outcome outcome = runScenario(ORANGE_PI, 1, 1.0, sleepMicros, 0, InputFrameTimePipeline::new);
        line(
                "input frame time proof: orderAndTimeErrors=%d of %d results, first: %s",
                outcome.orderAndTimeErrors(),
                outcome.checkedResults(),
                outcome.firstOrderAndTimeError());
        assertTrue(outcome.orderAndTimeErrors() > 0, "The wrong capture time was not found");
        assertTrue(
                outcome.firstOrderAndTimeError().contains("is not the time of the result frame"),
                outcome.firstOrderAndTimeError());
        // The tag data is still correct: only the time is wrong.
        assertEquals(outcome.checkedResults(), outcome.checkedEqual(), outcome.firstDifference());
    }

    /** A pipeline that fails after some calls. It records that it was released. */
    private static class FailingPipeline extends SimulatedPipeline {
        static final List<FailingPipeline> made = new java.util.concurrent.CopyOnWriteArrayList<>();
        int calls = 0;
        volatile boolean released = false;

        FailingPipeline(AprilTagPipelineSettings settings, int lanes, Model model) {
            super(settings, lanes, model);
            made.add(this);
        }

        @Override
        public CVPipelineResult run(Frame frame, QuirkyCamera cameraQuirks) {
            if (++calls > 100) {
                frame.release();
                throw new IllegalStateException("Simulated pipeline fault");
            }
            return super.run(frame, cameraQuirks);
        }

        @Override
        public void release() {
            released = true;
            super.release();
        }
    }

    /**
     * A scenario that a fault stops must put the test state back: no extra decode work for later
     * tests in the same JVM, and each pipeline and capture released. Needs PV_MLTAG_SIM=1.
     */
    @Test
    public void failedScenarioRestoresTheTestState() throws Exception {
        prepareSimulation(ORANGE_PI);
        assumeTrue(!NULL_PIPELINE, "The null pipeline has no pipeline that can fail");
        ensureReference(iterationsFor(1.0));
        long sleepMicros = calibrateCameraSleepMicros(CAMERA_FPS);
        FailingPipeline.made.clear();
        AssertionError error =
                assertThrows(
                        AssertionError.class,
                        () -> runScenario(ORANGE_PI, 2, 1.0, sleepMicros, 100, FailingPipeline::new));
        assertTrue(error.getMessage().contains("The vision loop failed"), error.getMessage());
        assertEquals(0, activeExtraDecodePercent, "The extra decode work is still on");
        assertEquals(2, FailingPipeline.made.size());
        for (FailingPipeline pipeline : FailingPipeline.made) {
            assertTrue(pipeline.released, "A pipeline was not released");
        }
        assertEquals(2, lastScenarioCaptures.size());
        for (GStreamerCapture capture : lastScenarioCaptures) {
            assertTrue(!capture.isOpen(), "A capture was not closed");
        }
    }
}
