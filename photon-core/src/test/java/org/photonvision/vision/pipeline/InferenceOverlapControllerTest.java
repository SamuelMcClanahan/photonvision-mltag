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

    /** A simulated vision loop. It models a camera that gives the newest frame. */
    private static class Simulation {
        final InferenceOverlapController controller;
        final int lanes;
        long cameraPeriod;
        long inference;
        long postInference;
        long loopOverhead = 300_000L;

        long now = 1_000 * MS;
        long lastFrameIndex = -1;
        // Completion times of the frames that are in inference, oldest first.
        final java.util.ArrayDeque<Long> pending = new java.util.ArrayDeque<>();
        final long[] laneFreeAt;
        double inferenceAverage = 0;
        int inferenceSamples = 0;

        int results = 0;
        int deferred = 0;
        int skipped = 0;
        int ownFrameResults = 0;

        Simulation(int lanes, long cameraPeriod, long inference, long postInference) {
            this.controller = new InferenceOverlapController(lanes);
            this.lanes = lanes;
            this.cameraPeriod = cameraPeriod;
            this.inference = inference;
            this.postInference = postInference;
            this.laneFreeAt = new long[lanes];
        }

        private long startInference() {
            int lane = 0;
            for (int i = 1; i < lanes; i++) {
                if (laneFreeAt[i] < laneFreeAt[lane]) {
                    lane = i;
                }
            }
            long done = Math.max(now, laneFreeAt[lane]) + inference;
            laneFreeAt[lane] = done;
            return done;
        }

        void step() {
            // Get the newest camera frame. Wait for it if no new frame is ready.
            long frameIndex = Math.max(lastFrameIndex + 1, now / cameraPeriod);
            long frameTime = frameIndex * cameraPeriod;
            now = Math.max(now, frameTime) + loopOverhead;
            lastFrameIndex = frameIndex;

            Action action =
                    controller.onFrame(now, frameTime, pending.size(), inferenceAverage, inferenceSamples);
            boolean ownFrame = false;
            if (action == Action.SKIP_INPUT) {
                assertFalse(pending.isEmpty(), "SKIP_INPUT needs a frame that is in inference");
                skipped++;
            } else {
                pending.addLast(startInference());
                if (action == Action.DEFER) {
                    assertTrue(pending.size() <= lanes - 1 + 1);
                    deferred++;
                    controller.onCallEnd(now, -1);
                    return;
                }
                ownFrame = pending.size() == 1;
            }

            long done = pending.removeFirst();
            now = Math.max(now, done);
            inferenceAverage =
                    inferenceSamples == 0 ? inference : inferenceAverage * 0.9 + inference * 0.1;
            inferenceSamples++;
            now += postInference;
            results++;
            if (ownFrame) {
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
        // RK3588: about 24 ms inference on one NPU core, 60 FPS camera.
        var serial = new Simulation(1, 16_666_667L, 24 * MS, 3 * MS);
        double serialFps = serial.run(600);
        assertTrue(serialFps < 40, "One lane is limited by the inference time: " + serialFps);

        var sim = new Simulation(2, 16_666_667L, 24 * MS, 3 * MS);
        sim.run(60);
        assertTrue(sim.controller.isOverlapActive(), "Overlap must start in less than 60 frames");
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5, "Two lanes must keep the camera rate");
        assertEquals(0, sim.skipped);
        assertEquals(1, sim.deferred, "Only the first overlap call gives no result");
    }

    @Test
    public void twoCamerasAtFiftyFramesEachNeedThreeLanesWorthOfNpu() {
        // One camera with 2 lanes at 50 FPS and a 30 ms model also keeps its rate.
        var sim = new Simulation(2, 20 * MS, 30 * MS, 3 * MS);
        sim.run(60);
        double fps = sim.run(500);
        assertEquals(50.0, fps, 0.5);
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
        assertEquals(1, sim.skipped, "One camera frame is skipped to go back to serial mode");

        int ownBefore = sim.ownFrameResults;
        int deferredBefore = sim.deferred;
        double fps = sim.run(600);
        assertEquals(600, sim.ownFrameResults - ownBefore);
        assertEquals(deferredBefore, sim.deferred);
        assertEquals(1, sim.skipped, "The modes must not change again and again");
        assertEquals(60.0, fps, 0.5);
    }

    @Test
    public void pipelineThatIsAlmostAsSlowAsTheCameraDoesNotChangeModeAgainAndAgain() {
        // The work is between 85% and 97% of the 16.7 ms frame period. Each of the two modes keeps
        // the camera rate here. The important property is that the mode is stable.
        for (long decode : new long[] {5_200_000L, 5_800_000L, 6_000_000L, 6_500_000L, 7_000_000L}) {
            var sim = new Simulation(2, 16_666_667L, 9 * MS, decode);
            sim.run(3000);
            assertTrue(sim.skipped <= 1, "At most one frame can be skipped: " + sim.skipped);
            assertTrue(sim.deferred <= 1, "The overlap can start one time only: " + sim.deferred);
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
        sim.run(200);
        assertTrue(sim.controller.isOverlapActive());
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5);
    }

    @Test
    public void threeLanesServeAVerySlowModel() {
        // 45 ms inference needs three lanes for a 60 FPS camera.
        var sim = new Simulation(3, 16_666_667L, 45 * MS, 2 * MS);
        sim.run(100);
        assertTrue(sim.controller.isOverlapActive());
        double fps = sim.run(600);
        assertEquals(60.0, fps, 0.5);
    }
}
