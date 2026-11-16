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

/**
 * Decides when a pipeline keeps more than one frame in model inference at the same time.
 *
 * <p>The pipeline has two modes:
 *
 * <ul>
 *   <li>SERIAL: each call completes its own frame. This gives the lowest latency. It is the correct
 *       mode when the pipeline is faster than the camera.
 *   <li>OVERLAP: a call starts the inference of its frame and completes an earlier frame. The
 *       inference of two (or more) frames then runs at the same time on different lanes. This is
 *       the correct mode when one inference takes longer than one camera frame period.
 * </ul>
 *
 * <p>The controller starts in SERIAL mode. It goes to OVERLAP mode when the vision thread has no
 * idle time between calls (the camera is faster than the pipeline) and the inference is the larger
 * part of the work. It goes back to SERIAL mode when the measured camera frame period shows that
 * the SERIAL mode can keep the camera rate. To go back, the pipeline must skip the input of one
 * camera frame for each frame that is in inference, because one call can give only one result.
 *
 * <p>All times are in nanoseconds. This class has no connection to frames or models, so a test can
 * give it simulated times.
 */
final class InferenceOverlapController {
    enum Action {
        /** Start the inference of the input frame, then complete the oldest frame. */
        COMPLETE_OLDEST,
        /** Start the inference of the input frame and give no result with this call. */
        DEFER,
        /** Do not use the input frame. Complete the oldest frame that is in inference. */
        SKIP_INPUT
    }

    /** Idle time below this value means that the vision thread does not wait for the camera. */
    static final long LOW_IDLE_NANOS = 1_500_000L;

    /** Number of calls in sequence without idle time before OVERLAP mode starts. */
    static final int CALLS_TO_START_OVERLAP = 8;

    /** Number of calls in sequence that show that SERIAL mode is sufficient before OVERLAP stops. */
    static final int CALLS_TO_STOP_OVERLAP = 60;

    /** A time between two frames above this value is a pause, not a frame period. */
    static final long MAX_FRAME_PERIOD_NANOS = 500_000_000L;

    private final int lanes;

    private boolean overlap = false;
    private boolean leavingOverlap = false;

    private long lastCallEndNanos = 0;
    private long lastFrameTimestampNanos = 0;
    private int lowIdleStreak = 0;
    private int serialSufficientStreak = 0;

    private double framePeriodAverage = 0;
    private int framePeriodSamples = 0;
    private double shortestRecentFramePeriod = 0;
    private int recentFramePeriodCount = 0;

    private double postInferenceAverage = 0;
    private int postInferenceSamples = 0;

    /**
     * The camera frame period that was measured in OVERLAP mode when the pipeline kept the camera
     * rate. 0 = not known. In SERIAL mode with no idle time the camera period cannot be measured,
     * because the pipeline then does not see each camera frame.
     */
    private double knownCameraPeriod = 0;

    InferenceOverlapController(int lanes) {
        this.lanes = Math.max(1, lanes);
    }

    boolean isOverlapActive() {
        return overlap;
    }

    /**
     * Call this at the start of a pipeline call.
     *
     * @param nowNanos The monotonic time now.
     * @param frameTimestampNanos The capture time of the input frame.
     * @param pendingFrames The number of frames that are in inference before this call.
     * @param inferenceAverageNanos The average time of one inference, or 0 if not known.
     * @param inferenceSamples The number of inference time samples.
     */
    Action onFrame(
            long nowNanos,
            long frameTimestampNanos,
            int pendingFrames,
            double inferenceAverageNanos,
            int inferenceSamples) {
        if (lanes <= 1) {
            return Action.COMPLETE_OLDEST;
        }

        long idleNanos = lastCallEndNanos > 0 ? nowNanos - lastCallEndNanos : -1;
        updateFramePeriod(frameTimestampNanos);

        double serialWorkNanos = inferenceAverageNanos + postInferenceAverage;

        if (leavingOverlap) {
            if (pendingFrames > 0) {
                return Action.SKIP_INPUT;
            }
            leavingOverlap = false;
            overlap = false;
            lowIdleStreak = 0;
        }

        if (!overlap) {
            // A camera that became faster makes the known period invalid.
            if (knownCameraPeriod > 0
                    && recentFramePeriodCount >= 8
                    && shortestRecentFramePeriod < 0.8 * knownCameraPeriod) {
                knownCameraPeriod = 0;
            }

            lowIdleStreak = idleNanos >= 0 && idleNanos < LOW_IDLE_NANOS ? lowIdleStreak + 1 : 0;
            boolean inferenceIsLargerPart =
                    inferenceSamples >= CALLS_TO_START_OVERLAP
                            && inferenceAverageNanos >= 0.5 * serialWorkNanos;
            boolean serialIsSufficient =
                    knownCameraPeriod > 0 && serialWorkNanos < 0.9 * knownCameraPeriod;
            if (lowIdleStreak >= CALLS_TO_START_OVERLAP && inferenceIsLargerPart && !serialIsSufficient) {
                overlap = true;
                serialSufficientStreak = 0;
                // The frame period from SERIAL mode is the pipeline period. Measure it again.
                framePeriodSamples = 0;
                framePeriodAverage = 0;
            }
        } else {
            boolean serialIsSufficient =
                    framePeriodSamples >= 8
                            && inferenceSamples > 0
                            && serialWorkNanos < 0.9 * framePeriodAverage;
            serialSufficientStreak = serialIsSufficient ? serialSufficientStreak + 1 : 0;
            if (serialSufficientStreak >= CALLS_TO_STOP_OVERLAP) {
                knownCameraPeriod = framePeriodAverage;
                serialSufficientStreak = 0;
                if (pendingFrames > 0) {
                    leavingOverlap = true;
                    return Action.SKIP_INPUT;
                }
                overlap = false;
                lowIdleStreak = 0;
            }
        }

        if (overlap && pendingFrames + 1 < lanes) {
            return Action.DEFER;
        }
        return Action.COMPLETE_OLDEST;
    }

    /**
     * Call this at the end of a pipeline call.
     *
     * @param nowNanos The monotonic time now.
     * @param postInferenceNanos The time of the work after the inference (tag decode and pose) in
     *     this call, or a negative value if this call completed no frame.
     */
    void onCallEnd(long nowNanos, long postInferenceNanos) {
        lastCallEndNanos = nowNanos;
        if (postInferenceNanos >= 0) {
            postInferenceAverage =
                    postInferenceSamples == 0
                            ? postInferenceNanos
                            : postInferenceAverage * 0.9 + postInferenceNanos * 0.1;
            postInferenceSamples++;
        }
    }

    /** Call this when the frames in inference were discarded, for example after new settings. */
    void reset() {
        overlap = false;
        leavingOverlap = false;
        lastCallEndNanos = 0;
        lastFrameTimestampNanos = 0;
        lowIdleStreak = 0;
        serialSufficientStreak = 0;
        framePeriodAverage = 0;
        framePeriodSamples = 0;
        shortestRecentFramePeriod = 0;
        recentFramePeriodCount = 0;
        knownCameraPeriod = 0;
    }

    private void updateFramePeriod(long frameTimestampNanos) {
        if (lastFrameTimestampNanos > 0 && frameTimestampNanos > lastFrameTimestampNanos) {
            long period = frameTimestampNanos - lastFrameTimestampNanos;
            if (period < MAX_FRAME_PERIOD_NANOS) {
                framePeriodAverage =
                        framePeriodSamples == 0 ? period : framePeriodAverage * 0.8 + period * 0.2;
                framePeriodSamples++;

                // Shortest period in a window of 30 frames.
                if (recentFramePeriodCount == 0 || recentFramePeriodCount >= 30) {
                    shortestRecentFramePeriod = period;
                    recentFramePeriodCount = 1;
                } else {
                    shortestRecentFramePeriod = Math.min(shortestRecentFramePeriod, period);
                    recentFramePeriodCount++;
                }
            } else {
                // A pause. The idle time before this call is not a sign of a fast pipeline.
                lowIdleStreak = 0;
            }
        }
        lastFrameTimestampNanos = frameTimestampNanos;
    }
}
