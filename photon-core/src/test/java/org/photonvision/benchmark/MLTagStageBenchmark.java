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

package org.photonvision.benchmark;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.awt.Color;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.ColorHelper;
import org.photonvision.common.util.TestUtils;
import org.photonvision.gstreamer.GStreamerCapture;
import org.photonvision.gstreamer.GStreamerCaptureJNI;
import org.photonvision.gstreamer.GStreamerFrame;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.objects.Letterbox;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeInput;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeParams;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagDetector;

/**
 * Stage timing for the ML AprilTag pipeline. It does not use a camera or an NPU. It measures the
 * CPU stages (JPEG decode, model input preparation, ROI decode) on recorded field pictures.
 *
 * <p>Run it on the coprocessor to get the stage times of that board:
 *
 * <pre>
 * PV_MLTAG_BENCH=1 PV_GSTREAMER_CAPTURE_JNI_LIB=/path/libpv_gstreamer_capture_jni.so \
 *   ./gradlew :photon-core:test --tests "*MLTagStageBenchmark*"
 * </pre>
 */
public class MLTagStageBenchmark {
    private static final int ROI_PADDING = 40;

    @TempDir static Path tempDir;

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    private static boolean enabled() {
        String raw = System.getenv("PV_MLTAG_BENCH");
        return raw != null && !raw.isBlank() && !raw.equals("0");
    }

    private static void line(String format, Object... args) {
        System.out.println("[bench] " + String.format(Locale.ROOT, format, args));
    }

    /** Median time of one call, in milliseconds. */
    private static double medianMillis(int warmup, int runs, Callable<?> body) throws Exception {
        for (int i = 0; i < warmup; i++) {
            body.call();
        }
        double[] samples = new double[runs];
        for (int i = 0; i < runs; i++) {
            long start = System.nanoTime();
            body.call();
            samples[i] = (System.nanoTime() - start) / 1e6;
        }
        Arrays.sort(samples);
        return samples[runs / 2];
    }

    private static Mat readField(String name) {
        Path path = TestUtils.getTestImagesPath(false).resolve("2024").resolve(name);
        Mat image = Imgcodecs.imread(path.toString(), Imgcodecs.IMREAD_COLOR);
        if (image.empty()) {
            throw new IllegalStateException("Could not read " + path);
        }
        return image;
    }

    /**
     * Makes a picture of the requested size from two field pictures. The first picture is scaled to
     * the full width, so the tags have the pixel size that a camera of this width gives for the same
     * field of view. The second picture fills the remaining rows.
     */
    private static Mat makeScene(String top, String bottom, int width, int height) {
        Mat scene = new Mat(height, width, CvType.CV_8UC3);
        Mat first = readField(top);
        int firstHeight =
                Math.min(height, (int) Math.round(first.rows() * (double) width / first.cols()));
        Mat firstScaled = new Mat();
        Imgproc.resize(first, firstScaled, new Size(width, firstHeight), 0, 0, Imgproc.INTER_CUBIC);
        firstScaled.copyTo(scene.submat(0, firstHeight, 0, width));
        if (firstHeight < height) {
            Mat second = readField(bottom);
            Mat secondScaled = new Mat();
            int secondHeight =
                    Math.max(
                            height - firstHeight,
                            (int) Math.round(second.rows() * (double) width / second.cols()));
            Imgproc.resize(
                    second, secondScaled, new Size(width, secondHeight), 0, 0, Imgproc.INTER_CUBIC);
            secondScaled
                    .submat(0, height - firstHeight, 0, width)
                    .copyTo(scene.submat(firstHeight, height, 0, width));
            second.release();
            secondScaled.release();
        }
        first.release();
        firstScaled.release();
        return scene;
    }

    private static AprilTagDetector newDetector(int threads) {
        AprilTagDetector detector = new AprilTagDetector();
        detector.addFamily(AprilTagFamily.kTag36h11.getNativeName());
        ROIDecodeParams defaults = new ROIDecodeParams();
        defaults.detectorConfig.numThreads = threads;
        defaults.detectorConfig.refineEdges = true;
        detector.setConfig(defaults.detectorConfig);
        detector.setQuadThresholdParameters(defaults.quadParams);
        return detector;
    }

    private static List<RotatedRect> tagRois(Mat gray) {
        AprilTagDetector detector = newDetector(4);
        List<RotatedRect> rois = new ArrayList<>();
        for (AprilTagDetection detection : detector.detect(gray)) {
            if (detection.getHamming() > 0 || detection.getDecisionMargin() < 35) {
                continue;
            }
            double minX = Double.MAX_VALUE;
            double maxX = -Double.MAX_VALUE;
            double minY = Double.MAX_VALUE;
            double maxY = -Double.MAX_VALUE;
            for (int i = 0; i < 4; i++) {
                minX = Math.min(minX, detection.getCornerX(i));
                maxX = Math.max(maxX, detection.getCornerX(i));
                minY = Math.min(minY, detection.getCornerY(i));
                maxY = Math.max(maxY, detection.getCornerY(i));
            }
            RotatedRect box =
                    new RotatedRect(
                            new Point((minX + maxX) / 2.0, (minY + maxY) / 2.0),
                            new Size(maxX - minX, maxY - minY),
                            0);
            rois.add(AprilTagROIDecodePipe.expandBbox(box, ROI_PADDING, gray.cols(), gray.rows()));
        }
        detector.close();
        return rois;
    }

    private static ROIDecodeParams pipeParams(int threads) {
        ROIDecodeParams params = new ROIDecodeParams();
        params.tagFamily = AprilTagFamily.kTag36h11;
        params.atrEnabled = false;
        params.detectorConfig.numThreads = threads;
        params.detectorConfig.refineEdges = true;
        return params;
    }

    /**
     * The decode as it was before: one ROI after the other, convert BGR to gray, 4-thread detector.
     */
    private static int previousRoiDecode(
            Mat bgr, List<RotatedRect> rois, AprilTagDetector detector, Mat grayScratch) {
        int found = 0;
        for (RotatedRect roi : rois) {
            Rect rect = roi.boundingRect();
            int x = Math.max(0, rect.x);
            int y = Math.max(0, rect.y);
            Rect clamped =
                    new Rect(
                            x, y, Math.min(rect.width, bgr.cols() - x), Math.min(rect.height, bgr.rows() - y));
            Mat sub = bgr.submat(clamped);
            Imgproc.cvtColor(sub, grayScratch, Imgproc.COLOR_BGR2GRAY);
            found += detector.detect(grayScratch).length;
            sub.release();
        }
        return found;
    }

    @Test
    public void fullFrameDetectorReference() throws Exception {
        assumeTrue(enabled(), "Set PV_MLTAG_BENCH=1 to run the benchmark");
        // The Rubik Pi 3 benchmark has this number: full-frame detect, 1280x800, 4 threads.
        Mat scene = makeScene("GeneralField1.jpg", "GeneralField3.jpg", 1280, 800);
        Mat gray = new Mat();
        Imgproc.cvtColor(scene, gray, Imgproc.COLOR_BGR2GRAY);
        for (int threads : new int[] {1, 4}) {
            AprilTagDetector detector = newDetector(threads);
            double ms = medianMillis(5, 25, () -> detector.detect(gray));
            line("full-frame detect 1280x800 threads=%d: %.2f ms", threads, ms);
            detector.close();
        }
    }

    @Test
    public void roiDecodeStage() throws Exception {
        assumeTrue(enabled(), "Set PV_MLTAG_BENCH=1 to run the benchmark");
        String[][] scenes = {
            {"GeneralField1.jpg", "GeneralField2.jpg"},
            {"GeneralField3.jpg", "GeneralField4.jpg"},
            {"SpeakerCenter_63in.jpg", "GeneralField5.jpg"},
            {"SpeakerCenter_19in.jpg", "Amp_42in.jpg"},
            {"Amp_85in.jpg", "SpeakerCenter_143in.jpg"},
        };

        AprilTagDetector previousDetector = newDetector(4);
        AprilTagROIDecodePipe newPipe = new AprilTagROIDecodePipe();
        newPipe.setParams(pipeParams(4));
        Mat grayScratch = new Mat();

        for (String[] names : scenes) {
            Mat bgr = makeScene(names[0], names[1], 1600, 1304);
            Mat gray = new Mat();
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
            List<RotatedRect> rois = tagRois(gray);
            long pixels = 0;
            int largest = 0;
            for (RotatedRect roi : rois) {
                Rect rect = roi.boundingRect();
                pixels += (long) rect.width * rect.height;
                largest = Math.max(largest, Math.max(rect.width, rect.height));
            }
            if (rois.isEmpty()) {
                line("scene %s: no tags", names[0]);
                continue;
            }

            double previous =
                    medianMillis(10, 60, () -> previousRoiDecode(bgr, rois, previousDetector, grayScratch));
            CVMat grayFrame = new CVMat(gray);
            double current = medianMillis(10, 60, () -> newPipe.run(new ROIDecodeInput(grayFrame, rois)));
            line(
                    "ROI decode 1600x1304 %-24s tags=%d roiPixels=%d largestSide=%d: previous=%.2f ms"
                            + " new=%.2f ms",
                    names[0], rois.size(), pixels, largest, previous, current);
            bgr.release();
            grayFrame.release();
        }
        previousDetector.close();
        newPipe.release();
    }

    @Test
    public void modelInputStage() throws Exception {
        assumeTrue(enabled(), "Set PV_MLTAG_BENCH=1 to run the benchmark");
        Size modelSize = new Size(640, 640);
        var pad = ColorHelper.colorToScalar(Color.GRAY);

        for (int[] size : new int[][] {{1280, 800}, {1600, 1304}}) {
            Mat bgr = makeScene("GeneralField1.jpg", "GeneralField2.jpg", size[0], size[1]);
            Mat gray = new Mat();
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);

            // Before: a new Mat for each frame, 3-channel resize, then BGR to RGB in the JNI.
            Mat rgb = new Mat();
            double previous =
                    medianMillis(
                            20,
                            200,
                            () -> {
                                Mat letterboxed = new Mat();
                                Letterbox.letterbox(bgr, letterboxed, modelSize, pad);
                                Imgproc.cvtColor(letterboxed, rgb, Imgproc.COLOR_BGR2RGB);
                                letterboxed.release();
                                return null;
                            });

            // Now: Mats are kept, 1-channel resize, gray to BGR, then BGR to RGB in the JNI.
            Mat scratch = new Mat();
            Mat letterboxed = new Mat();
            Mat expanded = new Mat();
            double current =
                    medianMillis(
                            20,
                            200,
                            () -> {
                                Letterbox.letterbox(gray, scratch, letterboxed, modelSize, pad);
                                Imgproc.cvtColor(letterboxed, expanded, Imgproc.COLOR_GRAY2BGR);
                                Imgproc.cvtColor(expanded, rgb, Imgproc.COLOR_BGR2RGB);
                                return null;
                            });
            line(
                    "model input %dx%d: previous(BGR)=%.3f ms new(gray)=%.3f ms",
                    size[0], size[1], previous, current);
        }
    }

    /** Frames per second of one capture pipeline that runs as fast as it can. */
    private static double pipelineFps(String pipeline, int outputMode, int frames) {
        try (GStreamerCapture capture = GStreamerCapture.open(pipeline, outputMode)) {
            if (capture == null) {
                return -1;
            }
            for (int i = 0; i < 10; i++) {
                GStreamerFrame frame = capture.grab(2000);
                if (frame != null) {
                    frame.toCVMat().release();
                }
            }
            long start = System.nanoTime();
            int got = 0;
            while (got < frames) {
                GStreamerFrame frame = capture.grab(2000);
                if (frame == null) {
                    return -1;
                }
                CVMat mat = frame.toCVMat();
                mat.release();
                got++;
            }
            return got / ((System.nanoTime() - start) / 1e9);
        }
    }

    @Test
    public void jpegDecodeStage() throws Exception {
        assumeTrue(enabled(), "Set PV_MLTAG_BENCH=1 to run the benchmark");
        assumeTrue(GStreamerCapture.isAvailable(), "Native GStreamer capture library is not loaded");
        assumeTrue(GStreamerCaptureJNI.hasCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE));

        for (int[] size : new int[][] {{1280, 800}, {1600, 1304}}) {
            Mat bgr = makeScene("GeneralField1.jpg", "GeneralField2.jpg", size[0], size[1]);
            // A monochrome camera sends a picture with no color: equal B, G and R.
            Mat gray = new Mat();
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
            Mat mono = new Mat();
            Imgproc.cvtColor(gray, mono, Imgproc.COLOR_GRAY2BGR);

            for (int quality : new int[] {75, 90}) {
                Path file = tempDir.resolve("bench-" + size[0] + "-" + quality + ".jpg");
                Imgcodecs.imwrite(
                        file.toString(),
                        mono,
                        new MatOfInt(
                                Imgcodecs.IMWRITE_JPEG_QUALITY,
                                quality,
                                Imgcodecs.IMWRITE_JPEG_SAMPLING_FACTOR,
                                Imgcodecs.IMWRITE_JPEG_SAMPLING_FACTOR_422));
                long bytes = Files.size(file);
                String source =
                        "multifilesrc location=\"" + file + "\" loop=true caps=image/jpeg,framerate=30/1 ! ";
                String sink =
                        "appsink name=pvappsink emit-signals=false max-buffers=1 drop=false sync=false";

                double legacy =
                        pipelineFps(
                                source + "jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! " + sink,
                                GStreamerCaptureJNI.OUTPUT_AUTO,
                                300);
                double directBgr = directDecodeMillis(source + sink, GStreamerCaptureJNI.OUTPUT_BGR);
                double directGray = directDecodeMillis(source + sink, GStreamerCaptureJNI.OUTPUT_GRAY);
                line(
                        "JPEG %dx%d q=%d (%d kB, 4:2:2 mono): legacy jpegdec+videoconvert=%.2f ms/frame"
                                + " directBGR=%.2f ms directGray=%.2f ms",
                        size[0],
                        size[1],
                        quality,
                        bytes / 1000,
                        legacy > 0 ? 1000.0 / legacy : -1,
                        directBgr,
                        directGray);
            }
        }
    }

    /** Average decode time of the native JPEG decoder, from its own counters. */
    private static double directDecodeMillis(String pipeline, int outputMode) throws Exception {
        try (GStreamerCapture capture = GStreamerCapture.open(pipeline, outputMode)) {
            if (capture == null) {
                return -1;
            }
            for (int i = 0; i < 20; i++) {
                GStreamerFrame frame = capture.grab(2000);
                if (frame != null) {
                    frame.toCVMat().release();
                }
            }
            GStreamerCapture.Stats before = capture.getStats();
            for (int i = 0; i < 300; i++) {
                GStreamerFrame frame = capture.grab(2000);
                if (frame != null) {
                    frame.toCVMat().release();
                }
            }
            GStreamerCapture.Stats after = capture.getStats();
            long frames = after.framesProduced() - before.framesProduced();
            return frames > 0 ? (after.convertNanosSum() - before.convertNanosSum()) / 1e6 / frames : -1;
        }
    }
}
