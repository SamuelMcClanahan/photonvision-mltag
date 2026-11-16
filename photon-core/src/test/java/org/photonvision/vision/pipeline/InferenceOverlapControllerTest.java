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
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;
import org.photonvision.vision.pipeline.InferenceOverlapController.Action;

/**
 * Tests the mode decisions with a simulated camera and a simulated pipeline. No frames, no model.
 */
public class InferenceOverlapControllerTest {
    private static final long MS = 1_000_000L;

    /**
     * A simulated pipeline: a camera that gives the newest frame, inference lanes, and one work
     * thread that does the work after the inference for one frame after the other.
     */
    private static class Simulation {
        final InferenceOverlapController controller;
        final int lanes;
        long cameraPeriod;
        long inference;
        long postInference;
        long loopOverhead = 300_000L;

        long now = 1_000 * MS;
        long lastFrameIndex = -1;
        // Times when the results of the frames in work are ready, oldest first.
        final java.util.ArrayDeque<Long> pending = new java.util.ArrayDeque<>();
        final long[] laneFreeAt;
        long workThreadFreeAt = 0;
        double inferenceAverage = 0;
        int inferenceSamples = 0;

        int results = 0;
        int deferred = 0;
        int skipped = 0;
        int ownFrameResults = 0;
        long waitedForResults = 0;

        Simulation(int lanes, long cameraPeriod, long inference, long postInference) {
            this(
                    lanes,
                    cameraPeriod,
                    inference,
                    postInference,
                    InferenceOverlapController.DEFAULT_RESERVE);
        }

        Simulation(int lanes, long cameraPeriod, long inference, long postInference, double reserve) {
            this.controller = new InferenceOverlapController(lanes, reserve);
            this.lanes = lanes;
            this.cameraPeriod = cameraPeriod;
            this.inference = inference;
            this.postInference = postInference;
            this.laneFreeAt = new long[lanes];
        }

        /** Starts a frame and returns the time when its result is ready. */
        private long startFrame() {
            int lane = 0;
            for (int i = 1; i < lanes; i++) {
                if (laneFreeAt[i] < laneFreeAt[lane]) {
                    lane = i;
                }
            }
            long inferenceDone = Math.max(now, laneFreeAt[lane]) + inference;
            laneFreeAt[lane] = inferenceDone;
            long ready = Math.max(inferenceDone, workThreadFreeAt) + postInference;
            workThreadFreeAt = ready;
            return ready;
        }

        void step() {
            // Get the newest camera frame. Wait for it if no new frame is ready.
            long frameIndex = Math.max(lastFrameIndex + 1, now / cameraPeriod);
            long frameTime = frameIndex * cameraPeriod;
            now = Math.max(now, frameTime) + loopOverhead;
            lastFrameIndex = frameIndex;

            int pendingBefore = pending.size();
            Action action =
                    controller.onFrame(now, frameTime, pendingBefore, inferenceAverage, inferenceSamples);
            if (action == Action.SKIP_INPUT) {
                assertFalse(pending.isEmpty(), "SKIP_INPUT needs a frame that is in work");
                skipped++;
            } else {
                pending.addLast(startFrame());
                if (action == Action.DEFER) {
                    assertTrue(pending.size() <= InferenceOverlapController.MAX_DEPTH);
                    deferred++;
                    controller.onCallEnd(now, -1);
                    return;
                }
            }

            long ready = pending.removeFirst();
            if (ready > now) {
                waitedForResults += ready - now;
                now = ready;
            }
            inferenceAverage =
                    inferenceSamples == 0 ? inference : inferenceAverage * 0.9 + inference * 0.1;
            inferenceSamples++;
            results++;
            if (pendingBefore == 0 && action == Action.COMPLETE_OLDEST) {
                ownFrameResults++;
            }
            controller.onCallEnd(now, postInference);
        }

        double run(int steps) {
            long start = now;
            int before = results;
            for (int i = 0; i < steps; i++) {
                step();
            }
            return (results - before) / ((now - start) / 1e9);
        }
    }

    @Test
    public void oneLaneAlwaysCompletesItsOwnFrame() {
        var sim = new Simulation(1, 16_666_667L, 30 * MS, 3 * MS);
        sim.run(200);
        assertEquals(200, sim.results);
        assertEquals(0, sim.deferred);
        assertFalse(sim.controller.isOverlapActive());
    }

    @Test
    public void fastPipelineStaysSerial() {
        // Rubik Pi 3: about 5 ms inference and 3 ms tag decode, 60 FPS camera.
        var sim = new Simulation(2, 16_666_667L, 5 * MS, 3 * MS);
        double fps = sim.run(600);
        assertFalse(sim.controller.isOverlapActive());
        assertEquals(0, sim.deferred);
        assertEquals(0, sim.skipped);
        assertEquals(600, sim.ownFrameResults, "Each call must give the result of its own frame");
        assertEquals(60.0, fps, 0.5);
    }

    @Test
    public void slowInferenceStartsOverlapAndReachesTheCameraRate() {
        // RK3588: about 22 ms inference on one NPU core and 8 ms tag decode, 60 FPS camera.
        var serial = new Simulation(1, 16_666_667L, 22 * MS, 8 * MS);
        double serialFps = serial.run(600);
        assertTrue(serialFps < 35, "One lane is limited by the work for one frame: " + serialFps);

        var sim = new Simulation(2, 16_666_667L, 22 * MS, 8 * MS);
        sim.run(120);
        assertTrue(sim.controller.isOverlapActive(), "Overlap must start in less than 120 frames");
        // 30 ms of work for a frame. Two frames in work give 33.3 ms. That is less than the
        // reserve of 0.3 frame periods (5 ms) above the work time, so three frames are in work.
        assertEquals(3, sim.controller.getDepth());
        long waitedBefore = sim.waitedForResults;
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5, "Two lanes must keep the camera rate");
        assertEquals(0, sim.skipped);
        assertEquals(3, sim.deferred, "Only the calls that increase the depth give no result");
        assertEquals(
                0, sim.waitedForResults - waitedBefore, "The vision thread must not wait for a result");
    }

    @Test
    public void workWithSufficientReserveUsesTwoFramesInWork() {
        // 26 ms of work for a frame. Two frames in work give 33.3 ms: the reserve is 7 ms.
        var sim = new Simulation(2, 16_666_667L, 22 * MS, 4 * MS);
        sim.run(120);
        assertTrue(sim.controller.isOverlapActive());
        assertEquals(2, sim.controller.getDepth());
        long waitedBefore = sim.waitedForResults;
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5);
        assertEquals(2, sim.deferred);
        assertEquals(0, sim.waitedForResults - waitedBefore);
    }

    @Test
    public void smallerReserveKeepsFewerFramesInWork() {
        // The same 30 ms of work with a reserve of -0.25: two frames in work (less latency).
        var sim = new Simulation(2, 16_666_667L, 22 * MS, 8 * MS, -0.25);
        sim.run(120);
        assertTrue(sim.controller.isOverlapActive());
        assertEquals(2, sim.controller.getDepth());
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5);
        assertEquals(2, sim.deferred);
    }

    @Test
    public void oneCameraAtFiftyFramesWithASlowerModelKeepsItsRate() {
        var sim = new Simulation(2, 20 * MS, 30 * MS, 3 * MS);
        sim.run(120);
        double fps = sim.run(500);
        assertEquals(50.0, fps, 0.5);
    }

    @Test
    public void depthDoesNotIncreaseWhenTheWorkThreadIsTheLimit() {
        // 20 ms of tag decode for each frame is more than the 16.7 ms frame period. More frames
        // in work cannot help, so the depth stays at the number of lanes (more depth only adds
        // latency).
        var sim = new Simulation(2, 16_666_667L, 22 * MS, 20 * MS);
        sim.run(600);
        assertEquals(2, sim.controller.getDepth());
        double fps = sim.run(600);
        assertEquals(50.0, fps, 1.0, "The rate is the rate of the work thread");
    }

    @Test
    public void overlapStopsWhenTheModelBecomesFast() {
        var sim = new Simulation(2, 16_666_667L, 24 * MS, 3 * MS);
        sim.run(100);
        assertTrue(sim.controller.isOverlapActive());

        // For example the first frames were slow, and now the model is fast.
        sim.inference = 5 * MS;
        sim.run(400);
        assertFalse(sim.controller.isOverlapActive(), "Serial mode gives the lowest latency");
        assertTrue(
                sim.skipped >= 1 && sim.skipped <= 2,
                "One camera frame is skipped for each frame in work to go back to serial mode");
        int skippedAfterChange = sim.skipped;

        int ownBefore = sim.ownFrameResults;
        int deferredBefore = sim.deferred;
        double fps = sim.run(600);
        assertEquals(600, sim.ownFrameResults - ownBefore);
        assertEquals(deferredBefore, sim.deferred);
        assertEquals(skippedAfterChange, sim.skipped, "The modes must not change again and again");
        assertEquals(60.0, fps, 0.5);
    }

    @Test
    public void pipelineThatIsAlmostAsSlowAsTheCameraDoesNotChangeModeAgainAndAgain() {
        // The work is between 85% and 97% of the 16.7 ms frame period. Each of the two modes keeps
        // the camera rate here. The important property is that the mode is stable.
        for (long decode : new long[] {5_200_000L, 5_800_000L, 6_000_000L, 6_500_000L, 7_000_000L}) {
            var sim = new Simulation(2, 16_666_667L, 9 * MS, decode);
            sim.run(3000);
            assertTrue(sim.skipped <= 2, "The overlap can stop one time only: " + sim.skipped);
            assertTrue(sim.deferred <= 2, "The overlap can start one time only: " + sim.deferred);
            double fps = sim.run(600);
            assertEquals(60.0, fps, 0.5);
        }
    }

    @Test
    public void pipelineWithSpareTimeStaysSerial() {
        // 8 ms inference + 5 ms decode = 78% of the frame period.
        var sim = new Simulation(2, 16_666_667L, 8 * MS, 5 * MS);
        double fps = sim.run(1200);
        assertFalse(sim.controller.isOverlapActive());
        assertEquals(0, sim.deferred);
        assertEquals(1200, sim.ownFrameResults);
        assertEquals(60.0, fps, 0.5);
    }

    @Test
    public void fasterCameraStartsOverlapAgain() {
        // 30 FPS camera: 20 ms inference is sufficient in serial mode.
        var sim = new Simulation(2, 33_333_333L, 20 * MS, 3 * MS);
        sim.run(300);
        assertFalse(sim.controller.isOverlapActive());

        // The camera mode changes to 60 FPS. Now one inference is longer than the frame period.
        sim.cameraPeriod = 16_666_667L;
        sim.lastFrameIndex = sim.now / sim.cameraPeriod;
        sim.run(300);
        assertTrue(sim.controller.isOverlapActive());
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5);
    }

    @Test
    public void pipelineThatDoesNotGetEachCameraFrameStaysInOverlap() {
        // 24 ms of work for a frame and a 60 FPS camera: the overlap is necessary. The vision thread
        // gets only a part of the camera frames (the times between its frames are one period, then
        // three periods, again and again), for example because the processor is overloaded. The mean
        // time between the frames is then 33 ms. That must not look like a slow camera, for which
        // one frame at a time would be sufficient.
        final long period = 16_666_667L;
        final double inference = 20 * MS;
        final long postInference = 4 * MS;
        var controller = new InferenceOverlapController(2);
        long now = 1_000 * MS;
        long frameTime = now;
        int pending = 0;
        int skipped = 0;

        // One frame at a time takes 24 ms for each call, so the vision thread has no idle time.
        for (int i = 0; i < 40 && !controller.isOverlapActive(); i++) {
            frameTime = Math.max(frameTime + period, now / period * period);
            now = Math.max(now, frameTime) + 300_000L;
            Action action = controller.onFrame(now, frameTime, pending, inference, i);
            if (action == Action.DEFER) {
                // The overlap starts: this call gives no result, and its frame stays in work.
                pending++;
                controller.onCallEnd(now, -1);
            } else {
                assertEquals(Action.COMPLETE_OLDEST, action);
                now += (long) inference + postInference;
                controller.onCallEnd(now, postInference);
            }
        }
        assertTrue(controller.isOverlapActive(), "The overlap must start");

        for (int i = 0; i < 600; i++) {
            frameTime += (i % 2 == 0) ? period : 3 * period;
            now = frameTime + 300_000L;
            Action action = controller.onFrame(now, frameTime, pending, inference, 100 + i);
            if (action == Action.DEFER) {
                pending++;
                controller.onCallEnd(now, -1);
            } else {
                if (action == Action.SKIP_INPUT) {
                    skipped++;
                    pending--;
                }
                controller.onCallEnd(now, postInference);
            }
        }
        assertTrue(controller.isOverlapActive(), "The overlap must stay active");
        assertEquals(0, skipped, "No camera frame can be skipped to go back to one frame at a time");
    }

    @Test
    public void threeLanesServeAVerySlowModel() {
        // 40 ms inference needs three lanes for a 60 FPS camera. 42 ms of work for a frame is less
        // than three frame periods (50 ms) minus the reserve (5 ms).
        var sim = new Simulation(3, 16_666_667L, 40 * MS, 2 * MS);
        sim.run(200);
        assertTrue(sim.controller.isOverlapActive());
        assertEquals(3, sim.controller.getDepth());
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5);
    }

    @Test
    public void serialControllerSkipsInputUntilNoFrameIsInWork() {
        // For example a reset while frames stay in work. SERIAL mode wants no frame in work, so
        // the controller must remove the frames in work before it completes the input frame.
        var controller = new InferenceOverlapController(2);
        long now = 1_000 * MS;
        long frameTime = now;
        assertEquals(Action.SKIP_INPUT, controller.onFrame(now, frameTime, 2, 20 * MS, 100));
        controller.onCallEnd(now + MS, 4 * MS);
        now += 200 * MS;
        frameTime += 200 * MS;
        assertEquals(Action.SKIP_INPUT, controller.onFrame(now, frameTime, 1, 20 * MS, 100));
        controller.onCallEnd(now + MS, 4 * MS);
        now += 200 * MS;
        frameTime += 200 * MS;
        assertEquals(Action.COMPLETE_OLDEST, controller.onFrame(now, frameTime, 0, 20 * MS, 100));
        assertFalse(controller.isOverlapActive());
    }

    @Test
    public void depthDecreasesAfterAShortWorkPeak() {
        // 25 ms of work for a frame: two frames in work.
        var sim = new Simulation(2, 16_666_667L, 22 * MS, 3 * MS);
        sim.run(120);
        assertEquals(2, sim.controller.getDepth());

        // One short peak of the tag decode time increases the depth.
        sim.postInference = 60 * MS;
        sim.run(3);
        sim.postInference = 3 * MS;
        sim.run(120);
        assertEquals(3, sim.controller.getDepth());

        // The work is again 25 ms, so the depth must go back to 2.
        int skippedBefore = sim.skipped;
        sim.run(1200);
        assertEquals(2, sim.controller.getDepth());
        assertEquals(skippedBefore + 1, sim.skipped, "One camera frame is skipped for the step down");

        int deferredBefore = sim.deferred;
        int skippedAfter = sim.skipped;
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5);
        assertEquals(deferredBefore, sim.deferred, "The depth must not change again and again");
        assertEquals(skippedAfter, sim.skipped, "The depth must not change again and again");
    }

    @Test
    public void depthDecreasesWhenTheCameraBecomesSlower() {
        // 32 ms of work at 60 FPS: three frames in work.
        var sim = new Simulation(2, 16_666_667L, 24 * MS, 8 * MS);
        sim.run(120);
        assertEquals(3, sim.controller.getDepth());

        // At 30 FPS two frames in work keep the camera rate with less latency.
        sim.cameraPeriod = 33_333_333L;
        sim.lastFrameIndex = sim.now / sim.cameraPeriod;
        sim.run(1500);
        assertTrue(sim.controller.isOverlapActive());
        assertEquals(2, sim.controller.getDepth());
    }

    @Test
    public void depthStaysAtTheLanesWhenTheWorkThreadIsTheLimit() {
        // 20 ms of tag decode is more than the frame period: the work thread gives 50 FPS with
        // two frames in work. More depth only adds latency.
        var sim = new Simulation(2, 16_666_667L, 26 * MS, 20 * MS);
        sim.run(600);
        assertEquals(2, sim.controller.getDepth());
        double fps = sim.run(600);
        assertEquals(50.0, fps, 1.0);
    }

    @Test
    public void depthStaysAtTheLanesWhenTheLanesAreTheLimit() {
        // 40 ms inference on two lanes gives 50 FPS with two frames in work.
        var sim = new Simulation(2, 16_666_667L, 40 * MS, 2 * MS);
        sim.run(600);
        assertEquals(2, sim.controller.getDepth());
        double fps = sim.run(600);
        assertEquals(50.0, fps, 1.0);
    }

    @Test
    public void depthDecreasesWhenTheWorkThreadBecomesTheLimit() {
        // 30 ms of work at 60 FPS: three frames in work, no stage is the limit.
        var sim = new Simulation(2, 16_666_667L, 22 * MS, 8 * MS);
        sim.run(120);
        assertEquals(3, sim.controller.getDepth());

        // The tag decode becomes longer than the frame period. The work thread now gives 50 FPS
        // with two frames in work, so the third frame only adds latency and must be removed.
        sim.postInference = 20 * MS;
        int skippedBefore = sim.skipped;
        sim.run(300);
        assertEquals(2, sim.controller.getDepth());
        assertEquals(skippedBefore + 1, sim.skipped, "One camera frame is skipped for the step down");

        double fps = sim.run(600);
        assertEquals(50.0, fps, 1.0);
    }

    @Test
    public void overlapStopsFastWhenTheCallRateBecomesLow() {
        final long period = 16_666_667L;
        final double inference = 20 * MS;
        final long postInference = 4 * MS;
        var controller = new InferenceOverlapController(2);
        long now = 1_000 * MS;
        long frameTime = now;
        int pending = 0;

        // One frame at a time takes 24 ms for each call, so the vision thread has no idle time.
        for (int i = 0; i < 40 && !controller.isOverlapActive(); i++) {
            frameTime = Math.max(frameTime + period, now / period * period);
            now = Math.max(now, frameTime) + 300_000L;
            Action action = controller.onFrame(now, frameTime, pending, inference, i);
            if (action == Action.DEFER) {
                pending++;
                controller.onCallEnd(now, -1);
            } else {
                assertEquals(Action.COMPLETE_OLDEST, action);
                now += (long) inference + postInference;
                controller.onCallEnd(now, postInference);
            }
        }
        assertTrue(controller.isOverlapActive(), "The overlap must start");

        // OVERLAP at the camera rate.
        for (int i = 0; i < 100; i++) {
            frameTime += period;
            now = frameTime + 300_000L;
            Action action = controller.onFrame(now, frameTime, pending, inference, 100 + i);
            if (action == Action.DEFER) {
                pending++;
                controller.onCallEnd(now, -1);
            } else {
                assertEquals(Action.COMPLETE_OLDEST, action);
                controller.onCallEnd(now, postInference);
            }
        }
        assertTrue(controller.isOverlapActive());
        int depth = controller.getDepth();
        assertEquals(depth, pending);

        // For example an FPS limit of 10: one call each 100 ms.
        int firstSkip = -1;
        int serialAt = -1;
        for (int i = 1; i <= 20 && serialAt < 0; i++) {
            frameTime += 100 * MS;
            now = frameTime + 300_000L;
            Action action = controller.onFrame(now, frameTime, pending, inference, 200 + i);
            if (action == Action.DEFER) {
                pending++;
                controller.onCallEnd(now, -1);
                continue;
            }
            if (action == Action.SKIP_INPUT) {
                pending--;
                if (firstSkip < 0) {
                    firstSkip = i;
                }
            } else if (pending == 0 && !controller.isOverlapActive()) {
                serialAt = i;
            }
            controller.onCallEnd(now + (long) inference + postInference, postInference);
        }
        assertTrue(firstSkip > 0 && firstSkip <= 5, "First skip: " + firstSkip);
        assertTrue(serialAt > 0 && serialAt <= 5 + depth, "Serial at: " + serialAt);
        assertFalse(controller.isOverlapActive());
    }

    @Test
    public void overlapStartsWhenTheTagDecodeIsTheLargerPart() {
        // 22 ms inference and 24 ms tag decode: SERIAL gives 21.7 FPS, OVERLAP gives 41.7 FPS.
        var sim = new Simulation(2, 16_666_667L, 22 * MS, 24 * MS);
        sim.run(600);
        assertTrue(sim.controller.isOverlapActive());
        assertEquals(2, sim.controller.getDepth());
        double fps = sim.run(600);
        assertEquals(1000.0 / 24, fps, 1.0);
    }

    @Test
    public void fastModelWithALongTagDecodeReachesTheCameraRate() {
        // 12 ms inference and 14 ms tag decode: SERIAL gives 38 FPS, OVERLAP gives 60 FPS.
        var sim = new Simulation(2, 16_666_667L, 12 * MS, 14 * MS);
        sim.run(600);
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5);
    }

    @Test
    public void fasterCameraMakesTheKnownPeriodInvalid() {
        // 43 ms of work at 30 FPS: OVERLAP.
        var sim = new Simulation(2, 33_333_333L, 40 * MS, 3 * MS);
        sim.run(200);
        assertTrue(sim.controller.isOverlapActive());

        // 29.5 ms of work: SERIAL keeps 30 FPS, the known camera period is 33.3 ms.
        sim.inference = 26_500_000L;
        sim.run(400);
        assertFalse(sim.controller.isOverlapActive());

        // The camera becomes faster (for example the FPS limit ends). SERIAL gets only a part of
        // the camera frames, so most of the frame times are still two periods.
        sim.cameraPeriod = 16_666_667L;
        sim.lastFrameIndex = sim.now / sim.cameraPeriod;
        sim.run(300);
        assertTrue(sim.controller.isOverlapActive());
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5);
    }
}
