How base-412ccea-roi-decode-pose.txt was made
=============================================

BaseEqualityTest (photon-core/src/test/java/org/photonvision/vision/pipe/) compares
the tag decode and the pose estimation of this build with this file. The file holds
the output of the code of the base commit 412ccea, which is before the speed work.

To make it again:
1. git worktree add <dir> 412ccea   (detached, outside the project work tree)
2. Put the class below in <dir>/photon-core/src/test/java/org/photonvision/vision/pipe/.
3. In <dir>: PV_BASE_GOLDEN_OUT=<file> ./gradlew :photon-core:test --tests "*BaseGoldenGenerator"
   -x spotlessApply -Porg.gradle.java.installations.paths=<JDK 25>
4. Copy <file> to test-resources/mltag/base-412ccea-roi-decode-pose.txt.
5. git worktree remove --force <dir>
The file was made on linux x86-64 with the WPILib/OpenCV natives of the base build.

Inputs: test-resources/testimages/2024/*.jpg and 2026/*.jpg (sorted by name) and
apriltag/36h11_stress_test.png, read with Imgcodecs.imread(IMREAD_COLOR).
Regions: the bounding box of each tag that a full-frame AprilTagDetector finds in the
gray picture (36h11, configuration and quad parameters of the base ROIDecodeParams,
no filter), made larger with AprilTagROIDecodePipe.expandBbox(box, 40, w, h), in the
order of the detector.
Decode: the base AprilTagROIDecodePipe with the default ROIDecodeParams, ATR on and
then ATR off, on the BGR picture with these regions. The base pipe gives the tags in
HashMap order, so the tags are sorted by ID.
Pose: the base AprilTagPoseEstimatorPipe as the base AprilTagPipeline uses it: tag size
6.5 in (TargetModel.kAprilTag6p5in_36h11), 40 iterations, a pinhole calibration of the
picture size with fx = fy = (w/2)/tan(40 deg), cx = w/2, cy = h/2, no distortion.
Format: all doubles are written with Double.toString, so they are exact.
  image <path> <w> <h> atr <on|off>
  calibration fx fy cx cy tagWidth
  region cx cy width height angle            (one line for each region)
  detection id hamming decisionMargin
  corners x0 y0 x1 y1 x2 y2 x3 y3
  center x y
  homography h0 .. h8
  pose1 x y z qw qx qy qz
  pose2 x y z qw qx qy qz
  errors error1 error2
  end

Source of the generator class (BaseGoldenGenerator.java):
---------------------------------------------------------
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

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
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
import org.wpilib.math.geometry.Transform3d;
import org.wpilib.math.util.Units;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagDetector;
import org.wpilib.vision.apriltag.AprilTagPoseEstimate;
import org.wpilib.vision.apriltag.AprilTagPoseEstimator;

/**
 * Temporary generator of the golden file for BaseEqualityTest. It runs only in a work tree of the
 * base commit 412ccea and is not committed there. Set PV_BASE_GOLDEN_OUT to the output file.
 */
public class BaseGoldenGenerator {
    static final int ROI_PADDING = 40;
    static final double HORIZONTAL_FOV_DEGREES = 80.0;
    static final int POSE_ITERATIONS = 40;

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    static List<Path> inputImages() throws IOException {
        Path root = TestUtils.getTestImagesPath(false);
        List<Path> images = new ArrayList<>();
        for (String folder : List.of("2024", "2026")) {
            try (Stream<Path> files = Files.list(root.resolve(folder))) {
                files.filter(p -> p.getFileName().toString().endsWith(".jpg"))
                        .sorted(Comparator.comparing(p -> p.getFileName().toString()))
                        .forEach(images::add);
            }
        }
        images.add(root.resolve("apriltag").resolve("36h11_stress_test.png"));
        return images;
    }

    static String d(double value) {
        return Double.toString(value);
    }

    static void transform(StringBuilder out, String name, Transform3d t) {
        var q = t.getRotation().getQuaternion();
        out.append(name)
                .append(' ')
                .append(d(t.getX()))
                .append(' ')
                .append(d(t.getY()))
                .append(' ')
                .append(d(t.getZ()))
                .append(' ')
                .append(d(q.getW()))
                .append(' ')
                .append(d(q.getX()))
                .append(' ')
                .append(d(q.getY()))
                .append(' ')
                .append(d(q.getZ()))
                .append('\n');
    }

    @Test
    public void generate() throws IOException {
        String outName = System.getenv("PV_BASE_GOLDEN_OUT");
        org.junit.jupiter.api.Assumptions.assumeTrue(outName != null, "Set PV_BASE_GOLDEN_OUT");
        StringBuilder out = new StringBuilder();
        out.append("# Golden ROI decode and pose data of base commit 412ccea. See README.txt.\n");

        ROIDecodeParams defaults = new ROIDecodeParams();
        // The tag size of TargetModel.kAprilTag6p5in_36h11, as the base AprilTagPipeline uses it.
        double tagWidth = Units.inchesToMeters(6.5);

        for (Path image : inputImages()) {
            Mat bgr = Imgcodecs.imread(image.toString(), Imgcodecs.IMREAD_COLOR);
            if (bgr.empty()) {
                throw new IllegalStateException("Could not read " + image);
            }
            Mat gray = new Mat();
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
            int width = bgr.cols();
            int height = bgr.rows();

            // (a) Regions: the boxes of the tags of a full-frame detector, in detector order.
            AprilTagDetector fullFrame = new AprilTagDetector();
            fullFrame.addFamily(defaults.tagFamily.getNativeName());
            fullFrame.setConfig(defaults.detectorConfig);
            fullFrame.setQuadThresholdParameters(defaults.quadParams);
            List<RotatedRect> regions = new ArrayList<>();
            for (AprilTagDetection det : fullFrame.detect(gray)) {
                double minX = Double.MAX_VALUE;
                double maxX = -Double.MAX_VALUE;
                double minY = Double.MAX_VALUE;
                double maxY = -Double.MAX_VALUE;
                for (int i = 0; i < 4; i++) {
                    minX = Math.min(minX, det.getCornerX(i));
                    maxX = Math.max(maxX, det.getCornerX(i));
                    minY = Math.min(minY, det.getCornerY(i));
                    maxY = Math.max(maxY, det.getCornerY(i));
                }
                RotatedRect box =
                        new RotatedRect(
                                new Point((minX + maxX) / 2.0, (minY + maxY) / 2.0),
                                new Size(maxX - minX, maxY - minY),
                                0);
                regions.add(AprilTagROIDecodePipe.expandBbox(box, ROI_PADDING, width, height));
            }
            fullFrame.close();

            // Pinhole calibration: 80 degree horizontal field of view, no distortion.
            double fx = (width / 2.0) / Math.tan(Math.toRadians(HORIZONTAL_FOV_DEGREES / 2.0));
            double fy = fx;
            double cx = width / 2.0;
            double cy = height / 2.0;
            var calibration =
                    new CameraCalibrationCoefficients(
                            new Size(width, height),
                            new JsonMatOfDouble(3, 3, new double[] {fx, 0, cx, 0, fy, cy, 0, 0, 1}),
                            new JsonMatOfDouble(1, 5, new double[] {0, 0, 0, 0, 0}),
                            new double[0],
                            List.of(),
                            new Size(0, 0),
                            0,
                            CameraLensModel.LENSMODEL_OPENCV);
            AprilTagPoseEstimatorPipe posePipe = new AprilTagPoseEstimatorPipe();
            posePipe.setParams(
                    new AprilTagPoseEstimatorPipeParams(
                            new AprilTagPoseEstimator.Config(tagWidth, fx, fy, cx, cy),
                            calibration,
                            POSE_ITERATIONS));

            for (boolean atr : new boolean[] {true, false}) {
                out.append("image ")
                        .append(image.getParent().getFileName())
                        .append('/')
                        .append(image.getFileName())
                        .append(' ')
                        .append(width)
                        .append(' ')
                        .append(height)
                        .append(" atr ")
                        .append(atr ? "on" : "off")
                        .append('\n');
                out.append("calibration ")
                        .append(d(fx))
                        .append(' ')
                        .append(d(fy))
                        .append(' ')
                        .append(d(cx))
                        .append(' ')
                        .append(d(cy))
                        .append(' ')
                        .append(d(tagWidth))
                        .append('\n');
                for (RotatedRect r : regions) {
                    out.append("region ")
                            .append(d(r.center.x))
                            .append(' ')
                            .append(d(r.center.y))
                            .append(' ')
                            .append(d(r.size.width))
                            .append(' ')
                            .append(d(r.size.height))
                            .append(' ')
                            .append(d(r.angle))
                            .append('\n');
                }

                // (b) The base ROI decode with its default parameters, on the BGR picture.
                ROIDecodeParams params = new ROIDecodeParams();
                params.atrEnabled = atr;
                AprilTagROIDecodePipe pipe = new AprilTagROIDecodePipe();
                pipe.setParams(params);
                CVMat frame = new CVMat(bgr);
                List<AprilTagDetection> detections =
                        new ArrayList<>(pipe.run(new ROIDecodeInput(frame, regions)).output);
                pipe.release();
                // The base pipe deduplicates with a HashMap. Sort by ID for a fixed order.
                detections.sort(Comparator.comparingInt(AprilTagDetection::getId));

                for (AprilTagDetection det : detections) {
                    out.append("detection ")
                            .append(det.getId())
                            .append(' ')
                            .append(det.getHamming())
                            .append(' ')
                            .append(d(det.getDecisionMargin()))
                            .append('\n');
                    out.append("corners");
                    for (int i = 0; i < 4; i++) {
                        out.append(' ').append(d(det.getCornerX(i))).append(' ').append(d(det.getCornerY(i)));
                    }
                    out.append('\n');
                    out.append("center ")
                            .append(d(det.getCenterX()))
                            .append(' ')
                            .append(d(det.getCenterY()))
                            .append('\n');
                    out.append("homography");
                    for (double h : det.getHomography()) {
                        out.append(' ').append(d(h));
                    }
                    out.append('\n');

                    // (c) The base single-tag pose estimation.
                    AprilTagPoseEstimate pose = posePipe.run(det).output;
                    transform(out, "pose1", pose.pose1);
                    transform(out, "pose2", pose.pose2);
                    out.append("errors ")
                            .append(d(pose.error1))
                            .append(' ')
                            .append(d(pose.error2))
                            .append('\n');
                }
                out.append("end\n");
            }
            posePipe.release();
            gray.release();
            bgr.release();
        }
        Files.writeString(Path.of(outName), out.toString(), StandardCharsets.UTF_8);
    }
}
