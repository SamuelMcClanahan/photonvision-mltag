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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeInput;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeParams;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagDetector;

/**
 * The parallel ROI decode must give the same tag data as a decode of each ROI, one after the other,
 * with a single-thread detector. These tests use field pictures that contain several tags.
 */
public class AprilTagROIParallelDecodeTest {
    private static final int ROI_PADDING = 40;

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    static List<Path> fieldImages() {
        Path folder = TestUtils.getTestImagesPath(false).resolve("2024");
        List<Path> images = new ArrayList<>();
        for (String name :
                List.of(
                        "GeneralField1.jpg",
                        "GeneralField2.jpg",
                        "GeneralField3.jpg",
                        "GeneralField4.jpg",
                        "GeneralField5.jpg",
                        "SpeakerCenter_63in.jpg",
                        "SpeakerCenter_143in.jpg",
                        "Amp_85in.jpg")) {
            images.add(folder.resolve(name));
        }
        return images;
    }

    static AprilTagDetector newDetector(int threads) {
        AprilTagDetector detector = new AprilTagDetector();
        detector.addFamily(AprilTagFamily.kTag36h11.getNativeName());
        // Use the settings that ROIDecodeParams and the AprilTag pipeline use.
        ROIDecodeParams defaults = new ROIDecodeParams();
        defaults.detectorConfig.numThreads = threads;
        defaults.detectorConfig.refineEdges = true;
        detector.setConfig(defaults.detectorConfig);
        detector.setQuadThresholdParameters(defaults.quadParams);
        return detector;
    }

    static ROIDecodeParams pipeParams(int threads) {
        ROIDecodeParams params = new ROIDecodeParams();
        params.tagFamily = AprilTagFamily.kTag36h11;
        params.maxHammingDistance = 0;
        params.minDecisionMargin = 35;
        params.atrEnabled = false;
        params.detectorConfig.numThreads = threads;
        params.detectorConfig.refineEdges = true;
        return params;
    }

    /** Simulates the boxes of the ML model: the bounding box of each tag that is in the picture. */
    static List<RotatedRect> roisFromFullFrameDetection(Mat gray) {
        AprilTagDetector detector = newDetector(1);
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

    /** One ROI after the other, single-thread detector, no shared state. This is the reference. */
    static Map<Integer, double[]> referenceDecode(Mat gray, List<RotatedRect> rois) {
        AprilTagDetector detector = newDetector(1);
        Map<Integer, double[]> best = new HashMap<>();
        Map<Integer, Float> bestMargin = new HashMap<>();
        for (RotatedRect roi : rois) {
            Rect rect = roi.boundingRect();
            int x = Math.max(0, rect.x);
            int y = Math.max(0, rect.y);
            int w = Math.min(rect.width, gray.cols() - x);
            int h = Math.min(rect.height, gray.rows() - y);
            Mat sub = gray.submat(new Rect(x, y, w, h));
            for (AprilTagDetection detection : detector.detect(sub)) {
                if (detection.getHamming() > 0 || detection.getDecisionMargin() < 35) {
                    continue;
                }
                Float previous = bestMargin.get(detection.getId());
                if (previous != null && detection.getDecisionMargin() <= previous) {
                    continue;
                }
                double[] corners = new double[8];
                for (int i = 0; i < 4; i++) {
                    corners[i * 2] = detection.getCornerX(i) + x;
                    corners[i * 2 + 1] = detection.getCornerY(i) + y;
                }
                best.put(detection.getId(), corners);
                bestMargin.put(detection.getId(), detection.getDecisionMargin());
            }
            sub.release();
        }
        detector.close();
        return best;
    }

    static void assertSameTagData(
            Map<Integer, double[]> expected, List<AprilTagDetection> actual, String message) {
        assertEquals(expected.size(), actual.size(), message + ": number of tags");
        for (AprilTagDetection detection : actual) {
            double[] corners = expected.get(detection.getId());
            assertNotNull(corners, message + ": tag " + detection.getId() + " is not in the reference");
            for (int i = 0; i < 4; i++) {
                // A different thread count can change the order of a floating-point sum. That gives a
                // difference near 1e-12 pixel. A real difference in the tag data is much larger.
                assertEquals(
                        corners[i * 2],
                        detection.getCornerX(i),
                        1e-6,
                        message + ": tag " + detection.getId() + " corner " + i + " x");
                assertEquals(
                        corners[i * 2 + 1],
                        detection.getCornerY(i),
                        1e-6,
                        message + ": tag " + detection.getId() + " corner " + i + " y");
            }
        }
    }

    @Test
    public void parallelDecodeGivesTheReferenceTagData() {
        int imagesWithSeveralTags = 0;
        int totalTags = 0;
        AprilTagROIDecodePipe parallelPipe = new AprilTagROIDecodePipe();
        parallelPipe.setParams(pipeParams(4));
        AprilTagROIDecodePipe serialPipe = new AprilTagROIDecodePipe();
        serialPipe.setParams(pipeParams(1));

        for (Path image : fieldImages()) {
            Mat gray = Imgcodecs.imread(image.toString(), Imgcodecs.IMREAD_GRAYSCALE);
            assertTrue(!gray.empty(), "Could not read " + image);
            List<RotatedRect> rois = roisFromFullFrameDetection(gray);
            if (rois.size() < 2) {
                gray.release();
                continue;
            }
            imagesWithSeveralTags++;
            totalTags += rois.size();

            Map<Integer, double[]> reference = referenceDecode(gray, rois);
            CVMat frame = new CVMat(gray);

            // Do each picture several times: thread timing must not change the result.
            for (int repeat = 0; repeat < 5; repeat++) {
                var parallel = parallelPipe.run(new ROIDecodeInput(frame, rois)).output;
                assertSameTagData(reference, parallel, image.getFileName() + " parallel");
            }
            var serial = serialPipe.run(new ROIDecodeInput(frame, rois)).output;
            assertSameTagData(reference, serial, image.getFileName() + " serial");

            // The parallel result and the serial result must also have the same order.
            var parallel = parallelPipe.run(new ROIDecodeInput(frame, rois)).output;
            assertEquals(serial.size(), parallel.size());
            for (int i = 0; i < serial.size(); i++) {
                assertEquals(serial.get(i).getId(), parallel.get(i).getId(), "detection order");
            }
            frame.release();
        }

        parallelPipe.release();
        serialPipe.release();
        assumeTrue(imagesWithSeveralTags > 0, "No test picture has two or more tags");
        assertTrue(totalTags >= 4, "The test pictures must contain several tags");
    }

    @Test
    public void colorFrameAndGrayFrameGiveTheSameTagData() {
        AprilTagROIDecodePipe pipe = new AprilTagROIDecodePipe();
        pipe.setParams(pipeParams(4));
        int checked = 0;

        for (Path image : fieldImages()) {
            Mat bgr = Imgcodecs.imread(image.toString(), Imgcodecs.IMREAD_COLOR);
            Mat gray = new Mat();
            Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY);
            List<RotatedRect> rois = roisFromFullFrameDetection(gray);
            if (rois.isEmpty()) {
                bgr.release();
                gray.release();
                continue;
            }
            checked++;

            Map<Integer, double[]> reference = referenceDecode(gray, rois);
            CVMat grayFrame = new CVMat(gray);
            CVMat colorFrame = new CVMat(bgr);
            assertSameTagData(
                    reference,
                    pipe.run(new ROIDecodeInput(grayFrame, rois)).output,
                    image.getFileName() + " gray frame");
            assertSameTagData(
                    reference,
                    pipe.run(new ROIDecodeInput(colorFrame, rois)).output,
                    image.getFileName() + " color frame");
            grayFrame.release();
            colorFrame.release();
        }

        pipe.release();
        assumeTrue(checked > 0, "No test picture has a tag");
    }

    @Test
    public void manyRoisAreDecodedInAStableOrder() {
        // More ROIs than workers, with duplicates. Duplicates must collapse to one detection.
        Path image = fieldImages().get(0);
        Mat gray = Imgcodecs.imread(image.toString(), Imgcodecs.IMREAD_GRAYSCALE);
        List<RotatedRect> base = roisFromFullFrameDetection(gray);
        assumeTrue(!base.isEmpty(), "The test picture has no tag");

        List<RotatedRect> rois = new ArrayList<>();
        for (int i = 0; i < 4; i++) {
            rois.addAll(base);
        }
        // Add ROIs that contain no tag.
        rois.add(new RotatedRect(new Point(60, 60), new Size(100, 100), 0));
        rois.add(new RotatedRect(new Point(gray.cols() - 70, 70), new Size(120, 120), 0));

        Map<Integer, double[]> reference = referenceDecode(gray, rois);
        AprilTagROIDecodePipe pipe = new AprilTagROIDecodePipe();
        pipe.setParams(pipeParams(4));
        CVMat frame = new CVMat(gray);

        List<Integer> firstOrder = null;
        for (int repeat = 0; repeat < 10; repeat++) {
            var result = pipe.run(new ROIDecodeInput(frame, rois)).output;
            assertSameTagData(reference, result, "repeat " + repeat);
            List<Integer> order = new ArrayList<>();
            for (AprilTagDetection detection : result) {
                order.add(detection.getId());
            }
            if (firstOrder == null) {
                firstOrder = order;
            } else {
                assertEquals(firstOrder, order, "The detection order must not change");
            }
        }
        assertTrue(
                firstOrder.stream().sorted(Comparator.naturalOrder()).distinct().count()
                        == firstOrder.size());

        frame.release();
        pipe.release();
    }
}
