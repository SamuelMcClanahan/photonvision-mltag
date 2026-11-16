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
        // in work cannot help, so the depth must stay small (more depth only adds latency).
        var sim = new Simulation(2, 16_666_667L, 22 * MS, 20 * MS);
        sim.run(600);
        assertTrue(sim.controller.getDepth() <= 3, "Depth: " + sim.controller.getDepth());
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
}
