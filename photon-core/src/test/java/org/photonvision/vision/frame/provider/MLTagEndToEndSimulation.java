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
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.awt.Color;
import java.lang.management.ManagementFactory;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
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
 * each frame. Each result of a load run must be equal to the recorded data of its frame.
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
 * <p>The run scripts set PV_OPENCV_THREADS=1 and PV_GSTREAMER_CAPTURE_RT_PRIORITY=10. Use the same
 * settings for the simulation. The real-time priority needs root or CAP_SYS_NICE.
 *
 * <p>The report gives for each camera: the mean result rate (resultFps), the rate that 99% of the
 * times between two results are better than (p99FrameFps), the lowest value in 100 of the FPS value
 * that PhotonVision shows, which is a mean of 20 frames (p99ReportedFps), and the lowest rate in a
 * window of one second (worstSecondFps).
 *
 * <pre>
 * PV_MLTAG_SIM=1 PV_MLTAG_SIM_MODE=work PV_MLTAG_SIM_CAMERAS=2 PV_MLTAG_SIM_SECONDS=30 \
 *   PV_OPENCV_THREADS=1 PV_GSTREAMER_CAPTURE_RT_PRIORITY=10 \
 *   PV_GSTREAMER_CAPTURE_JNI_LIB=/path/libpv_gstreamer_capture_jni.so \
 *   ./gradlew :photon-core:test --tests "*MLTagEndToEndSimulation.orangePi5"
 * </pre>
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

    @TempDir static Path tempDir;

    /**
     * A coprocessor profile.
     *
     * @param name Name for the report.
     * @param npuCores Number of inferences that the NPU can run at the same time.
     * @param inferenceMillis Mean time of the detect call of the NPU library for one frame.
     * @param inferenceSpread Standard deviation of the inference time, as a part of the mean.
     * @param lanes Value of PV_MLTAG_INFERENCE_LANES in the run script of the board.
     */
    private record Board(
            String name, int npuCores, double inferenceMillis, double inferenceSpread, int lanes) {}

    /**
     * Rubik Pi 3 (QCS6490). The AprilTag benchmark of 2026-07-09 measured 5.1 ms for the complete
     * model stage, including about 0.5 ms of input preparation that the simulation does itself.
     * Qualcomm AI Hub gives 3.7 to 4.1 ms for YOLOv11 detection (w8a8) on this NPU. One HTP runs one
     * graph at a time.
     */
    private static final Board RUBIK = new Board("Rubik Pi 3", 1, 4.6, 0.0, 1);

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
    private static final Board ORANGE_PI = new Board("Orange Pi 5", 3, 23.0, 0.1025, 2);

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
     * @param jpegBytes Size of the Motion-JPEG frame.
     * @param boxes The regions that the simulated model gives for this frame.
     */
    private record SceneFrame(int source, long jpegBytes, List<RotatedRect> boxes) {}

    private static SceneFrame[] sceneFrames;
    private static List<String> sourceNames;
    private static Path frameDirectory;

    /**
     * The tag data of each frame from a calm run: a slow camera and a pipeline that completes each
     * frame before the next frame arrives. Each result of a load run must be equal to this data.
     */
    private static List<List<double[]>> referenceTags;

    /** Number of frames that the simulated model could not identify. It must stay 0. */
    private static final AtomicInteger unknownFrames = new AtomicInteger();

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

    private static Path framePath(int index) {
        return frameDirectory.resolve(String.format(Locale.ROOT, "frame_%04d.jpg", index));
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
            long bytes = writeJpeg(scene, framePath(index));
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

        prepareReference();

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
            minBytes = Math.min(minBytes, frame.jpegBytes());
            maxBytes = Math.max(maxBytes, frame.jpegBytes());
            sumBytes += frame.jpegBytes();
            minBoxes = Math.min(minBoxes, frame.boxes().size());
            maxBoxes = Math.max(maxBoxes, frame.boxes().size());
            sumBoxes += frame.boxes().size();
            int tags = referenceTags.get(i).size();
            minTags = Math.min(minTags, tags);
            maxTags = Math.max(maxTags, tags);
            sumTags += tags;
        }
        double count = sceneFrames.length;
        line(
                "frame set: %d different frames from %d real pictures (%d frames for each picture),"
                        + " %dx%d monochrome MJPEG 4:2:2. Made in %.0f s.",
                sceneFrames.length,
                names.size(),
                framesForEachSource,
                WIDTH,
                HEIGHT,
                (System.nanoTime() - start) / 1e9);
        line(
                "frame set: JPEG size min/mean/max = %d/%.0f/%d kB | model regions for each frame"
                        + " min/mean/max = %d/%.1f/%d | decoded tags for each frame min/mean/max ="
                        + " %d/%.1f/%d",
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
                        tags += referenceTags.get(i).size();
                        bytes += sceneFrames[i].jpegBytes();
                    }
                }
                line(
                        "  source %2d %-44s regions=%.1f (%s) decodedTags=%.1f jpegKb=%d",
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
     * Compares the tag data of a result with the reference data of the same frame.
     *
     * @param withPose Compare the pose too. Not possible when the load run uses more pose iterations
     *     than the reference run.
     * @return null if the data is equal, or a text that tells the difference.
     */
    private static String compareWithReference(int index, List<double[]> tags, boolean withPose) {
        List<double[]> expected = referenceTags.get(index);
        if (expected.size() != tags.size()) {
            return "frame " + index + ": " + tags.size() + " tags, reference " + expected.size();
        }
        int values = withPose ? 15 : 9;
        for (int tag = 0; tag < tags.size(); tag++) {
            for (int i = 0; i < values; i++) {
                if (Math.abs(expected.get(tag)[i] - tags.get(tag)[i]) > 1e-9) {
                    return "frame "
                            + index
                            + " tag "
                            + (int) expected.get(tag)[0]
                            + " value "
                            + i
                            + ": "
                            + tags.get(tag)[i]
                            + ", reference "
                            + expected.get(tag)[i];
                }
            }
        }
        return null;
    }

    /**
     * Makes the reference data: a slow camera gives each frame, and a pipeline with one lane and no
     * NPU wait completes each frame before the next frame arrives. The frames go through the same
     * capture library and the same JPEG decoder as in the load runs.
     */
    private static void prepareReference() throws Exception {
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
        AprilTagPipeline pipeline =
                new SimulatedPipeline(
                        makeSettings(1.0), 1, new SimulatedModel(new SimulatedNpu(1, 0, 0.0, 0), 1.0));
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
                    reference.set(index, tags);
                    missing--;
                } else {
                    // The same frame must always give the same data.
                    List<List<double[]>> before = referenceTags;
                    referenceTags = reference;
                    String difference = compareWithReference(index, tags, true);
                    referenceTags = before;
                    assertTrue(difference == null, "The reference run is not stable: " + difference);
                }
                result.release();
            }
        } finally {
            pipeline.release();
            camera.release();
        }
        assertEquals(0, missing, "The reference run did not get each frame");
        assertEquals(0, unknownFrames.get(), "The simulated model could not identify a frame");
        referenceTags = reference;
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
        return "multifilesrc location=\""
                + frameDirectory.resolve("frame_%04d.jpg")
                + "\" index="
                + firstFrame
                + " start-index=0 stop-index="
                + (sceneFrames.length - 1)
                + " loop=true caps=image/jpeg,framerate=60/1 ! identity name=pvcapturetime sleep-time="
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
                    frame.toCaptureMetadata(
                            FrameCaptureMetadata.SOURCE_GSTREAMER_JNI, MathUtils.wpiNanoTime()));
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
                private final Mat resizedScratch = new Mat();
                private final Mat letterboxed = new Mat();
                private final Mat letterboxedBgr = new Mat();
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
                        Letterbox.letterbox(
                                in,
                                resizedScratch,
                                letterboxed,
                                new Size(640, 640),
                                ColorHelper.colorToScalar(Color.GRAY));
                        Mat modelInput = letterboxed;
                        if (letterboxed.channels() == 1) {
                            Imgproc.cvtColor(letterboxed, letterboxedBgr, Imgproc.COLOR_GRAY2BGR);
                            modelInput = letterboxedBgr;
                        }
                        // The JNI libraries convert BGR to RGB.
                        Imgproc.cvtColor(modelInput, rgb, Imgproc.COLOR_BGR2RGB);
                    }

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
                    resizedScratch.release();
                    letterboxed.release();
                    letterboxedBgr.release();
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
        settings.numIterations = (int) Math.round(40 * workMultiplier);
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

    /** The loop of VisionRunner for one camera, with measurement. */
    private static class CameraLoop implements Runnable {
        final SimulatedCamera camera;
        final AprilTagPipeline pipeline;
        final AtomicBoolean running = new AtomicBoolean(true);
        final AtomicBoolean measuring = new AtomicBoolean(false);
        final boolean comparePose;

        long results = 0;
        long resultsEqualToReference = 0;
        String firstDifference;
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

        CameraLoop(SimulatedCamera camera, AprilTagPipeline pipeline, boolean comparePose) {
            this.camera = camera;
            this.pipeline = pipeline;
            this.comparePose = comparePose;
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
                        // variation of the simulated camera and the JPEG decode alone.
                        long taken = System.nanoTime();
                        if (measuring.get()) {
                            results++;
                            resultsEqualToReference++;
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
                    if (measuring.get()) {
                        pipeNanos += (long) result.processingNanos;
                        results++;
                        // The tag data must be equal to the data of the calm run of this frame.
                        int index = readFrameIndex(result.inputAndOutputFrame.colorImage.getMat());
                        String difference =
                                index < 0
                                        ? "a result frame has no valid index"
                                        : compareWithReference(index, tagData(result), comparePose);
                        if (difference == null) {
                            resultsEqualToReference++;
                        } else if (firstDifference == null) {
                            firstDifference = difference;
                        }
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
            double[] reportedFpsLow) {}

    /**
     * Runs one scenario.
     *
     * @param board The coprocessor profile.
     * @param cameras Number of cameras.
     * @param slowdown The CPU of the coprocessor is this many times slower than this computer.
     * @param sleepMicros Camera pacing from {@link #calibrateCameraSleepMicros(double)}.
     */
    private static Outcome runScenario(Board board, int cameras, double slowdown, long sleepMicros)
            throws Exception {
        // Time scaling: all times of this computer are 1 / slowdown of the coprocessor times.
        // Work mode: real time, and the CPU stages do "slowdown" times their work.
        final double timeScale = WORK_MODE ? 1.0 : slowdown;
        final double workMultiplier = WORK_MODE ? slowdown : 1.0;
        GStreamerCaptureJNI.setTestExtraDecodePercent(
                WORK_MODE ? (int) Math.round((slowdown - 1.0) * 100) : 0);
        // The CPU part of the inference is CPU work: it does not get shorter with the time scale,
        // and it is multiplied in the work mode.
        SimulatedNpu npu =
                new SimulatedNpu(
                        board.npuCores(),
                        Math.round(board.inferenceMillis() * 1e6 / timeScale),
                        board.inferenceSpread(),
                        Math.round(NPU_CPU_MILLIS * 1e6 * workMultiplier));

        // PV_MLTAG_SIM_APP_CPUS=2,3 does what "taskset -c 2,3 java" does for the application: the
        // threads that start from here get these CPUs. Use it with
        // PV_GSTREAMER_CAPTURE_DECODE_CPUS, which gives each decode thread its own CPU.
        String appCpus = System.getenv("PV_MLTAG_SIM_APP_CPUS");
        if (appCpus != null && !appCpus.isBlank()) {
            long mask = 0;
            for (int cpu : intList("PV_MLTAG_SIM_APP_CPUS", new int[0])) {
                mask |= 1L << cpu;
            }
            assertTrue(GStreamerCaptureJNI.setCurrentThreadCpuMask(mask), "Could not set the CPU mask");
        }

        var properties = new FrameStaticProperties(WIDTH, HEIGHT, 80.0, makeCalibration());
        List<CameraLoop> loops = new ArrayList<>();
        List<Thread> threads = new ArrayList<>();
        List<GStreamerCapture> captures = new ArrayList<>();
        for (int i = 0; i < cameras; i++) {
            // Each camera starts at a different frame of the set, so the cameras do not have the
            // same view at the same time.
            int firstFrame = (int) ((long) i * sceneFrames.length / cameras);
            GStreamerCapture capture =
                    GStreamerCapture.open(
                            cameraPipeline(sleepMicros, firstFrame), GStreamerCaptureJNI.OUTPUT_GRAY);
            assertNotNull(capture, "Could not open the simulated camera");
            captures.add(capture);
            AprilTagPipeline pipeline =
                    NULL_PIPELINE
                            ? null
                            : new SimulatedPipeline(
                                    makeSettings(workMultiplier),
                                    board.lanes(),
                                    new SimulatedModel(npu, workMultiplier));
            // The pose of the reference run has the standard number of iterations.
            var loop =
                    new CameraLoop(new SimulatedCamera(capture, properties), pipeline, workMultiplier == 1.0);
            loops.add(loop);
            Thread thread = new Thread(loop, "SimVision-" + i);
            threads.add(thread);
            thread.start();
        }

        var os =
                (com.sun.management.OperatingSystemMXBean) ManagementFactory.getOperatingSystemMXBean();

        // Warm-up: JIT compile, detector creation, start of the inference overlap.
        Thread.sleep(5000);
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

        for (CameraLoop loop : loops) {
            loop.running.set(false);
        }
        for (Thread thread : threads) {
            thread.join(5000);
        }

        double[] resultFps = new double[cameras];
        double[] p99Fps = new double[cameras];
        double[] worstSecondFps = new double[cameras];
        double[] reportedFpsLow = new double[cameras];
        List<Long> allLatency = new ArrayList<>();
        long results = 0;
        long complete = 0;
        String firstDifference = null;
        long getNanos = 0;
        long runNanos = 0;
        long pipeNanos = 0;
        for (int i = 0; i < cameras; i++) {
            CameraLoop loop = loops.get(i);
            getNanos += loop.getNanos;
            runNanos += loop.runNanos;
            pipeNanos += loop.pipeNanos;
            if (loop.failure != null) {
                throw new AssertionError("The vision loop failed", loop.failure);
            }
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
            complete += loop.resultsEqualToReference;
            if (firstDifference == null) {
                firstDifference = loop.firstDifference;
            }
            if (loop.pipeline != null) {
                loop.pipeline.release();
            }
            loop.camera.release();
        }
        if (System.getenv("PV_MLTAG_SIM_DETAIL") != null) {
            // Times between results, in coprocessor milliseconds: how many are in each range.
            int[] limits = {18, 20, 22, 25, 30, 40, 1000};
            int[] counts = new int[limits.length];
            int total = 0;
            // Long times between results (25 ms or more), and how many of them have a pipeline call
            // that waited 3 ms or more for the result (the result was not ready). The other long
            // times come from a late camera frame.
            int longIntervals = 0;
            int longBecauseOfWait = 0;
            for (CameraLoop loop : loops) {
                for (int i = 1; i < loop.resultTimesNanos.size(); i++) {
                    double ms =
                            (loop.resultTimesNanos.get(i) - loop.resultTimesNanos.get(i - 1)) / 1e6 * timeScale;
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
                                (loop.resultTimesNanos.get(i) - loop.resultTimesNanos.get(i - 1)) / 1e6 * timeScale;
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

        GStreamerCaptureJNI.setTestExtraDecodePercent(0);

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
                (double) complete / results,
                firstDifference,
                decodeMillis * timeScale,
                cpuCores,
                getNanos / 1e6 / results * timeScale,
                runNanos / 1e6 / results * timeScale,
                pipeNanos / 1e6 / results * timeScale,
                p99Fps,
                worstSecondFps,
                reportedFpsLow);
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
        line(
                "%-11s cameras=%d cpuSlowdown=%.1f npuMs=%.1f | cameraFps=%.1f resultFps=%s"
                        + "p99FrameFps=%sp99ReportedFps=%sworstSecondFps=%s| latencyMs p50=%.1f p95=%.1f max=%.1f | decodeMs=%.1f getMs=%.1f runMs=%.1f"
                        + " pipesMs=%.1f | equalToReference=%.2f%% | cpuCoresBusy=%.2f",
                board.name(),
                cameras,
                slowdown,
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
                o.cpuCores());
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

    private static void runBoard(Board defaultBoard) throws Exception {
        assumeTrue(enabled(), "Set PV_MLTAG_SIM=1 to run the simulation");
        assumeTrue(GStreamerCapture.isAvailable(), "Native GStreamer capture library is not loaded");
        assumeTrue(GStreamerCaptureJNI.hasCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE));
        prepareScene();
        line(
                "mode: %s. ATR (adaptive tag resizing): %s. CPU time of the NPU library: %.1f ms for"
                        + " each inference. Computer: %d CPUs.",
                WORK_MODE ? "real time with more CPU work" : "time scaling",
                ATR_ENABLED ? "on (pipeline default)" : "off",
                NPU_CPU_MILLIS,
                Runtime.getRuntime().availableProcessors());

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
            long sleepMicros = calibrateCameraSleepMicros(WORK_MODE ? CAMERA_FPS : CAMERA_FPS * slowdown);
            for (double inferenceMillis : inferenceTimes) {
                Board board =
                        new Board(
                                defaultBoard.name(),
                                defaultBoard.npuCores(),
                                inferenceMillis,
                                defaultBoard.inferenceSpread(),
                                defaultBoard.lanes());
                for (int cameras : intList("PV_MLTAG_SIM_CAMERAS", new int[] {1, 2})) {
                    Outcome outcome = runScenario(board, cameras, slowdown, sleepMicros);
                    report(board, cameras, slowdown, outcome);
                    // The tag data of each result must be equal to the data of the calm run of the
                    // same frame, also at full load.
                    assertEquals(0, unknownFrames.get(), "The simulated model could not identify a frame");
                    assertEquals(
                            1.0,
                            outcome.equalToReferenceFraction(),
                            1e-12,
                            "A result is not equal to the reference: " + outcome.firstDifference());
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
}
