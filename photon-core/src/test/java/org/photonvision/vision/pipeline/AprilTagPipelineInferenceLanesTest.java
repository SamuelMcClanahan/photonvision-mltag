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

package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Family;
import org.photonvision.common.configuration.NeuralNetworkModelManager.Version;
import org.photonvision.common.configuration.NeuralNetworkModelsSettings.ModelProperties;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.objects.ObjectDetector;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TrackedTarget;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagDetector;

/**
 * Tests for the ML AprilTag pipeline with more than one frame in model inference at the same time.
 * A simulated model replaces the NPU: it waits for a set time and then gives the boxes of the tags.
 */
public class AprilTagPipelineInferenceLanesTest {
    private static final double FRAME_PERIOD_MS = 1000.0 / 60.0;

    private static Path imagePath;
    private static List<RotatedRect> tagBoxes;

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();

        imagePath = TestUtils.getTestImagesPath(false).resolve("2024").resolve("GeneralField1.jpg");
        Mat gray = Imgcodecs.imread(imagePath.toString(), Imgcodecs.IMREAD_GRAYSCALE);
        AprilTagDetector detector = new AprilTagDetector();
        detector.addFamily(AprilTagFamily.kTag36h11.getNativeName());
        // Use the quad settings of the AprilTag pipeline.
        var decodeDefaults =
                new org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe.ROIDecodeParams();
        detector.setConfig(decodeDefaults.detectorConfig);
        detector.setQuadThresholdParameters(decodeDefaults.quadParams);
        tagBoxes = new ArrayList<>();
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
            tagBoxes.add(
                    new RotatedRect(
                            new Point((minX + maxX) / 2.0, (minY + maxY) / 2.0),
                            new Size(maxX - minX, maxY - minY),
                            0));
        }
        detector.close();
        gray.release();
        assertTrue(tagBoxes.size() >= 2, "The test picture must contain two or more tags");
    }

    /** A model that takes a fixed time and gives the boxes of the tags in the test picture. */
    private static class SimulatedModel implements Model {
        volatile long inferenceMillis;
        final AtomicInteger loads = new AtomicInteger();
        final AtomicInteger running = new AtomicInteger();
        final AtomicInteger maxRunning = new AtomicInteger();
        final AtomicInteger inferences = new AtomicInteger();

        SimulatedModel(long inferenceMillis) {
            this.inferenceMillis = inferenceMillis;
        }

        @Override
        public ObjectDetector load() {
            loads.incrementAndGet();
            Model model = this;
            return new ObjectDetector() {
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
                    int now = running.incrementAndGet();
                    maxRunning.accumulateAndGet(now, Math::max);
                    try {
                        // The frame must be valid during the full inference.
                        assertFalse(in.empty(), "The frame was released during the inference");
                        Thread.sleep(inferenceMillis);
                        assertFalse(in.empty(), "The frame was released during the inference");
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                    } finally {
                        running.decrementAndGet();
                    }
                    inferences.incrementAndGet();
                    List<NeuralNetworkPipeResult> results = new ArrayList<>();
                    for (RotatedRect box : tagBoxes) {
                        results.add(new NeuralNetworkPipeResult(box, 0, 0.9));
                    }
                    return results;
                }

                @Override
                public void release() {}
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

    private static class TestPipeline extends AprilTagPipeline {
        private final Model model;

        TestPipeline(int lanes, Model model) {
            super(makeSettings(), lanes);
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

    private static AprilTagPipelineSettings makeSettings() {
        var settings = new AprilTagPipelineSettings();
        settings.useMLDetection = true;
        settings.atrEnabled = false;
        settings.tagFamily = AprilTagFamily.kTag36h11;
        settings.solvePNPEnabled = false;
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

    private static FileFrameProvider newProvider(AprilTagPipeline pipeline) {
        var provider = new FileFrameProvider(imagePath, TestUtils.WPI2020Image.FOV, 1000, null);
        provider.requestFrameThresholdType(pipeline.getThresholdType());
        return provider;
    }

    private record Output(long sequenceId, long captureNanos, List<double[]> corners) {}

    private static Output toOutput(CVPipelineResult result) {
        List<double[]> corners = new ArrayList<>();
        List<TrackedTarget> targets = new ArrayList<>(result.targets);
        targets.sort((a, b) -> Integer.compare(a.getFiducialId(), b.getFiducialId()));
        for (TrackedTarget target : targets) {
            double[] values = new double[9];
            values[0] = target.getFiducialId();
            var points = target.getTargetCorners();
            for (int i = 0; i < 4; i++) {
                values[1 + i * 2] = points.get(i).x;
                values[2 + i * 2] = points.get(i).y;
            }
            corners.add(values);
        }
        return new Output(result.sequenceID, result.getImageCaptureTimestampNanos(), corners);
    }

    /** What one run of the simulated camera loop gave. */
    private static class RunStats {
        final List<Output> outputs = new ArrayList<>();
        final List<Boolean> ownFrame = new ArrayList<>();
        int callsWithoutResult = 0;
        long elapsedNanos = 0;

        double fps() {
            return outputs.size() / (elapsedNanos / 1e9);
        }
    }

    /**
     * Runs the pipeline as the vision thread does it, with a simulated 60 FPS camera. The camera
     * gives the newest frame: if the pipeline is slow, camera frames are skipped.
     */
    private static RunStats runAtCameraRate(
            AprilTagPipeline pipeline, FileFrameProvider provider, int calls)
            throws InterruptedException {
        RunStats stats = new RunStats();
        long periodNanos = (long) (FRAME_PERIOD_MS * 1e6);
        long start = System.nanoTime();
        long lastFrameIndex = -1;
        for (int i = 0; i < calls; i++) {
            long frameIndex = Math.max(lastFrameIndex + 1, (System.nanoTime() - start) / periodNanos);
            long wait = start + frameIndex * periodNanos - System.nanoTime();
            if (wait > 0) {
                Thread.sleep(wait / 1_000_000, (int) (wait % 1_000_000));
            }
            lastFrameIndex = frameIndex;

            Frame frame = provider.get();
            CVPipelineResult result = pipeline.run(frame, QuirkyCamera.DefaultCamera);
            if (result == null) {
                stats.callsWithoutResult++;
                continue;
            }
            // The capture time must be the time of the frame that the result is for.
            assertEquals(
                    result.inputAndOutputFrame.timestampNanos, result.getImageCaptureTimestampNanos());
            assertEquals(result.inputAndOutputFrame.sequenceID, result.sequenceID);
            stats.ownFrame.add(result.inputAndOutputFrame == frame);
            stats.outputs.add(toOutput(result));
            result.release();
        }
        stats.elapsedNanos = System.nanoTime() - start;
        return stats;
    }

    @Test
    public void twoLanesGiveTheSameTagDataInTheSameOrder() throws InterruptedException {
        var reference = new TestPipeline(1, new SimulatedModel(1));
        var referenceProvider = newProvider(reference);
        List<Output> expected = runAtCameraRate(reference, referenceProvider, 4).outputs;
        reference.release();
        referenceProvider.release();
        assertEquals(4, expected.size());
        assertEquals(tagBoxes.size(), expected.get(0).corners().size(), "All tags must decode");

        // One inference takes 30 ms. That is longer than the 16.7 ms camera frame period.
        var model = new SimulatedModel(30);
        var pipeline = new TestPipeline(2, model);
        var provider = newProvider(pipeline);
        RunStats stats = runAtCameraRate(pipeline, provider, 80);
        assertEquals(2, model.loads.get(), "Each lane must have its own copy of the model");
        assertEquals(2, model.maxRunning.get(), "Two frames must be in inference together");
        assertEquals(1, stats.callsWithoutResult, "Only the first overlap call gives no result");
        pipeline.release();
        provider.release();

        long previousSequence = -1;
        long previousCapture = -1;
        for (Output output : stats.outputs) {
            assertTrue(output.sequenceId() > previousSequence, "Results must be in the capture order");
            assertTrue(output.captureNanos() > previousCapture, "Capture times must increase");
            previousSequence = output.sequenceId();
            previousCapture = output.captureNanos();

            assertEquals(expected.get(0).corners().size(), output.corners().size());
            for (int tag = 0; tag < output.corners().size(); tag++) {
                double[] want = expected.get(0).corners().get(tag);
                double[] got = output.corners().get(tag);
                for (int i = 0; i < want.length; i++) {
                    assertEquals(want[i], got[i], 1e-6, "tag data must be equal");
                }
            }
        }
    }

    @Test
    public void twoLanesKeepTheCameraRateWhenTheModelIsSlow() throws InterruptedException {
        // One lane: 30 ms for each frame. The loop then gets each second camera frame.
        var oneLaneModel = new SimulatedModel(30);
        var oneLane = new TestPipeline(1, oneLaneModel);
        var oneLaneProvider = newProvider(oneLane);
        runAtCameraRate(oneLane, oneLaneProvider, 10);
        double oneLaneFps = runAtCameraRate(oneLane, oneLaneProvider, 60).fps();
        oneLane.release();
        oneLaneProvider.release();

        var twoLaneModel = new SimulatedModel(30);
        var twoLanes = new TestPipeline(2, twoLaneModel);
        var twoLaneProvider = newProvider(twoLanes);
        // Warm-up: the pipeline starts with one frame at a time and then starts the overlap.
        runAtCameraRate(twoLanes, twoLaneProvider, 30);
        double twoLaneFps = runAtCameraRate(twoLanes, twoLaneProvider, 120).fps();
        twoLanes.release();
        twoLaneProvider.release();

        System.out.println(
                "[lanes] simulated 30 ms model, 60 FPS camera: 1 lane = "
                        + String.format("%.1f", oneLaneFps)
                        + " FPS, 2 lanes = "
                        + String.format("%.1f", twoLaneFps)
                        + " FPS");
        assertEquals(1, oneLaneModel.maxRunning.get());
        assertEquals(2, twoLaneModel.maxRunning.get(), "Two frames must be in inference together");
        assertTrue(oneLaneFps < 36, "One lane cannot be faster than the model: " + oneLaneFps);
        assertTrue(twoLaneFps > 50, "Two lanes must keep near the 60 FPS camera rate: " + twoLaneFps);
    }

    @Test
    public void fastModelCompletesEachFrameImmediately() throws InterruptedException {
        // 3 ms is much less than the camera frame period. Overlap would only add latency.
        var model = new SimulatedModel(3);
        var pipeline = new TestPipeline(2, model);
        var provider = newProvider(pipeline);
        RunStats stats = runAtCameraRate(pipeline, provider, 60);
        pipeline.release();
        provider.release();

        assertEquals(60, stats.outputs.size(), "Each call must give the result of its own frame");
        assertEquals(0, stats.callsWithoutResult);
        assertFalse(stats.ownFrame.contains(false), "No result can be for an earlier frame");
        assertEquals(1, model.maxRunning.get());
    }

    @Test
    public void overlapStopsWhenTheModelBecomesFast() throws InterruptedException {
        var model = new SimulatedModel(30);
        var pipeline = new TestPipeline(2, model);
        var provider = newProvider(pipeline);
        RunStats slow = runAtCameraRate(pipeline, provider, 40);
        assertEquals(1, slow.callsWithoutResult);
        assertFalse(slow.ownFrame.get(slow.ownFrame.size() - 1), "The overlap must be active");

        model.inferenceMillis = 3;
        // The pipeline goes back to one frame at a time after about 60 frames at the camera rate.
        runAtCameraRate(pipeline, provider, 150);
        RunStats fast = runAtCameraRate(pipeline, provider, 40);
        pipeline.release();
        provider.release();

        assertEquals(40, fast.outputs.size());
        assertFalse(fast.ownFrame.contains(false), "Each call must complete its own frame again");
    }

    @Test
    public void releaseWaitsForFramesThatAreInInference() throws InterruptedException {
        var model = new SimulatedModel(40);
        var pipeline = new TestPipeline(2, model);
        var provider = newProvider(pipeline);

        List<Frame> frames = new ArrayList<>();
        Frame lastInput = null;
        for (int i = 0; i < 12; i++) {
            Frame frame = provider.get();
            frames.add(frame);
            lastInput = frame;
            CVPipelineResult result = pipeline.run(frame, QuirkyCamera.DefaultCamera);
            if (result != null) {
                result.release();
            }
            Thread.sleep(17);
        }
        assertNotNull(lastInput);

        // The simulated model checks that its frame is valid until the inference ends.
        pipeline.release();
        for (Frame frame : frames) {
            assertTrue(frame.colorImage.isReleased(), "Each frame must be released exactly one time");
        }
        assertSame(lastInput, frames.get(frames.size() - 1));
        provider.release();
    }
}
