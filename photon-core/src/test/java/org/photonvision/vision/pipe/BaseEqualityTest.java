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

package org.photonvision.vision.pipe;

import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.concurrent.Future;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.calibration.JsonMatOfDouble;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe.AprilTagPoseEstimatorPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeInput;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeParams;
import org.photonvision.vision.pipe.impl.VisionWorkerPool;
import org.wpilib.math.geometry.Transform3d;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagPoseEstimate;
import org.wpilib.vision.apriltag.AprilTagPoseEstimator;

/**
 * Proves that the tag decode and the pose estimation of this build give EXACTLY the data of the
 * base commit 412ccea, which is before the speed work. The golden file was made with the base code
 * (see test-resources/mltag/README.txt). Each variant gets the same picture and the same regions as
 * the base run. The doubles are compared bit for bit.
 */
public class BaseEqualityTest {
    /** Threads of the pipeline for the parallel variants. It gives parallel ROI groups. */
    private static final int PARALLEL_THREADS = 4;

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    /** The golden data of one detection. */
    private record GoldenDetection(
            int id,
            int hamming,
            double margin,
            double[] corners,
            double[] center,
            double[] homography,
            double[] pose1,
            double[] pose2,
            double[] errors) {}

    /** The golden data of one picture and one ATR mode. */
    private record GoldenCase(
            String image,
            int width,
            int height,
            boolean atr,
            double[] calibration,
            List<RotatedRect> regions,
            List<GoldenDetection> detections) {}

    private static Path resources() {
        return TestUtils.getResourcesFolderPath(false);
    }

    private static double[] numbers(String[] parts, int from) {
        double[] values = new double[parts.length - from];
        for (int i = from; i < parts.length; i++) {
            values[i - from] = Double.parseDouble(parts[i]);
        }
        return values;
    }

    private static List<GoldenCase> readGolden() throws IOException {
        Path file = resources().resolve("mltag").resolve("base-412ccea-roi-decode-pose.txt");
        List<GoldenCase> cases = new ArrayList<>();
        String image = null;
        int width = 0;
        int height = 0;
        boolean atr = false;
        double[] calibration = null;
        List<RotatedRect> regions = null;
        List<GoldenDetection> detections = null;
        String[] detection = null;
        double[] corners = null;
        double[] center = null;
        double[] homography = null;
        double[] pose1 = null;
        double[] pose2 = null;
        for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
            if (line.isBlank() || line.startsWith("#")) {
                continue;
            }
            String[] p = line.trim().split(" ");
            switch (p[0]) {
                case "image" -> {
                    image = p[1];
                    width = Integer.parseInt(p[2]);
                    height = Integer.parseInt(p[3]);
                    atr = p[5].equals("on");
                    regions = new ArrayList<>();
                    detections = new ArrayList<>();
                }
                case "calibration" -> calibration = numbers(p, 1);
                case "region" -> {
                    double[] r = numbers(p, 1);
                    regions.add(new RotatedRect(new Point(r[0], r[1]), new Size(r[2], r[3]), r[4]));
                }
                case "detection" -> detection = p;
                case "corners" -> corners = numbers(p, 1);
                case "center" -> center = numbers(p, 1);
                case "homography" -> homography = numbers(p, 1);
                case "pose1" -> pose1 = numbers(p, 1);
                case "pose2" -> pose2 = numbers(p, 1);
                case "errors" ->
                        detections.add(
                                new GoldenDetection(
                                        Integer.parseInt(detection[1]),
                                        Integer.parseInt(detection[2]),
                                        Double.parseDouble(detection[3]),
                                        corners,
                                        center,
                                        homography,
                                        pose1,
                                        pose2,
                                        numbers(p, 1)));
                case "end" ->
                        cases.add(
                                new GoldenCase(image, width, height, atr, calibration, regions, detections));
                default -> throw new IllegalStateException("Unknown line in the golden file: " + line);
            }
        }
        return cases;
    }

    private static ROIDecodeParams decodeParams(boolean atr, int threads) {
        // The base default parameters. Only the thread count selects the parallel path.
        ROIDecodeParams params = new ROIDecodeParams();
        params.atrEnabled = atr;
        params.detectorConfig.numThreads = threads;
        return params;
    }

    private static List<AprilTagDetection> decode(
            Mat image, List<RotatedRect> regions, boolean atr, int threads) {
        AprilTagROIDecodePipe pipe = new AprilTagROIDecodePipe();
        pipe.setParams(decodeParams(atr, threads));
        CVMat frame = new CVMat(image, null, false);
        List<AprilTagDetection> result =
                new ArrayList<>(pipe.run(new ROIDecodeInput(frame, regions)).output);
        pipe.release();
        // The golden file is sorted by ID, because the base pipe gives the HashMap order.
        result.sort(Comparator.comparingInt(AprilTagDetection::getId));
        return result;
    }

    private static AprilTagPoseEstimatorPipeParams poseParams(GoldenCase c) {
        double fx = c.calibration()[0];
        double fy = c.calibration()[1];
        double cx = c.calibration()[2];
        double cy = c.calibration()[3];
        double tagWidth = c.calibration()[4];
        var calibration =
                new CameraCalibrationCoefficients(
                        new Size(c.width(), c.height()),
                        new JsonMatOfDouble(3, 3, new double[] {fx, 0, cx, 0, fy, cy, 0, 0, 1}),
                        new JsonMatOfDouble(1, 5, new double[] {0, 0, 0, 0, 0}),
                        new double[0],
                        List.of(),
                        new Size(0, 0),
                        0,
                        CameraLensModel.LENSMODEL_OPENCV);
        return new AprilTagPoseEstimatorPipeParams(
                new AprilTagPoseEstimator.Config(tagWidth, fx, fy, cx, cy), calibration, 40);
    }

    /**
     * The present parallel pose path of AprilTagPipeline: the tags are put into groups (tag i in
     * group i % groupCount), each group has its own pose pipe, group 0 runs on the calling thread and
     * the other groups run on the shared worker pool.
     */
    private static AprilTagPoseEstimate[] parallelPoses(
            List<AprilTagDetection> detections, AprilTagPoseEstimatorPipeParams params)
            throws Exception {
        AprilTagPoseEstimate[] poses = new AprilTagPoseEstimate[detections.size()];
        if (detections.isEmpty()) {
            return poses;
        }
        int groupCount =
                Math.max(1, Math.min(detections.size(), VisionWorkerPool.maxGroups(PARALLEL_THREADS)));
        params.calibration().getCameraIntrinsicsMat();
        params.calibration().getDistCoeffsMat();
        List<AprilTagPoseEstimatorPipe> pipes = new ArrayList<>();
        for (int g = 0; g < groupCount; g++) {
            var pipe = new AprilTagPoseEstimatorPipe();
            pipe.setParams(params);
            pipes.add(pipe);
        }
        List<Future<?>> futures = new ArrayList<>();
        for (int g = 1; g < groupCount; g++) {
            final int group = g;
            futures.add(
                    VisionWorkerPool.EXECUTOR.submit(
                            () -> poseGroup(detections, pipes.get(group), group, groupCount, poses)));
        }
        poseGroup(detections, pipes.get(0), 0, groupCount, poses);
        for (Future<?> future : futures) {
            future.get();
        }
        for (var pipe : pipes) {
            pipe.release();
        }
        return poses;
    }

    private static void poseGroup(
            List<AprilTagDetection> detections,
            AprilTagPoseEstimatorPipe pipe,
            int group,
            int groupCount,
            AprilTagPoseEstimate[] poses) {
        for (int i = group; i < detections.size(); i += groupCount) {
            poses[i] = pipe.run(detections.get(i)).output;
        }
    }

    private static void same(
            List<String> mismatches, String where, String field, double golden, double actual) {
        if (Double.doubleToLongBits(golden) != Double.doubleToLongBits(actual)) {
            mismatches.add(
                    where + " " + field + ": golden=" + golden + " actual=" + actual);
        }
    }

    private static double[] transform(Transform3d t) {
        var q = t.getRotation().getQuaternion();
        return new double[] {t.getX(), t.getY(), t.getZ(), q.getW(), q.getX(), q.getY(), q.getZ()};
    }

    private static void compare(
            List<String> mismatches,
            String where,
            List<GoldenDetection> golden,
            List<AprilTagDetection> actual,
            AprilTagPoseEstimate[] poses) {
        if (golden.size() != actual.size()) {
            List<Integer> actualIds = actual.stream().map(AprilTagDetection::getId).toList();
            List<Integer> goldenIds = golden.stream().map(GoldenDetection::id).toList();
            mismatches.add(where + " tag IDs: golden=" + goldenIds + " actual=" + actualIds);
            return;
        }
        for (int i = 0; i < golden.size(); i++) {
            GoldenDetection g = golden.get(i);
            AprilTagDetection a = actual.get(i);
            String tag = where + " tag " + g.id();
            if (g.id() != a.getId() || g.hamming() != a.getHamming()) {
                mismatches.add(
                        tag + " id/hamming: golden=" + g.id() + "/" + g.hamming() + " actual="
                                + a.getId() + "/" + a.getHamming());
                continue;
            }
            same(mismatches, tag, "decision margin", g.margin(), a.getDecisionMargin());
            for (int k = 0; k < 4; k++) {
                same(mismatches, tag, "corner " + k + " x", g.corners()[k * 2], a.getCornerX(k));
                same(mismatches, tag, "corner " + k + " y", g.corners()[k * 2 + 1], a.getCornerY(k));
            }
            same(mismatches, tag, "center x", g.center()[0], a.getCenterX());
            same(mismatches, tag, "center y", g.center()[1], a.getCenterY());
            double[] h = a.getHomography();
            for (int k = 0; k < 9; k++) {
                same(mismatches, tag, "homography " + k, g.homography()[k], h[k]);
            }
            if (poses != null) {
                double[] p1 = transform(poses[i].pose1);
                double[] p2 = transform(poses[i].pose2);
                for (int k = 0; k < 7; k++) {
                    same(mismatches, tag, "pose1[" + k + "]", g.pose1()[k], p1[k]);
                    same(mismatches, tag, "pose2[" + k + "]", g.pose2()[k], p2[k]);
                }
                same(mismatches, tag, "error1", g.errors()[0], poses[i].error1);
                same(mismatches, tag, "error2", g.errors()[1], poses[i].error2);
            }
        }
    }

    @Test
    public void decodeAndPoseAreEqualToBase() throws Exception {
        List<GoldenCase> cases = readGolden();
        assertTrue(cases.size() >= 2, "The golden file has no data");
        List<String> mismatches = new ArrayList<>();
        int tags = 0;

        for (GoldenCase c : cases) {
            Path file = TestUtils.getTestImagesPath(false).resolve(c.image());
            Mat bgr = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_COLOR);
            assertTrue(!bgr.empty(), "Could not read " + file);
            assertTrue(bgr.cols() == c.width() && bgr.rows() == c.height(), "Size of " + file);
            Mat gray = new Mat();
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
            String where = c.image() + " atr=" + (c.atr() ? "on" : "off");
            var poseParams = poseParams(c);
            tags += c.detections().size();

            var serialBgr = decode(bgr, c.regions(), c.atr(), 1);
            compare(mismatches, where + " [serial decode, BGR]", c.detections(), serialBgr, null);

            var parallelBgr = decode(bgr, c.regions(), c.atr(), PARALLEL_THREADS);
            compare(
                    mismatches,
                    where + " [parallel decode, BGR, parallel pose]",
                    c.detections(),
                    parallelBgr,
                    parallelBgr.size() == c.detections().size()
                            ? parallelPoses(parallelBgr, poseParams)
                            : null);

            var parallelGray = decode(gray, c.regions(), c.atr(), PARALLEL_THREADS);
            compare(
                    mismatches,
                    where + " [parallel decode, gray, parallel pose]",
                    c.detections(),
                    parallelGray,
                    parallelGray.size() == c.detections().size()
                            ? parallelPoses(parallelGray, poseParams)
                            : null);

            bgr.release();
            gray.release();
        }

        System.out.println(
                "[base-equality] cases="
                        + cases.size()
                        + " goldenTags="
                        + tags
                        + " mismatches="
                        + mismatches.size());
        for (int i = 0; i < Math.min(40, mismatches.size()); i++) {
            System.out.println("[base-equality] " + mismatches.get(i));
        }
        assertTrue(tags >= 10, "The golden file must contain several tags");
        assertTrue(
                mismatches.isEmpty(),
                mismatches.size()
                        + " values differ from the base commit. First: "
                        + mismatches.subList(0, Math.min(10, mismatches.size())));
    }
}
