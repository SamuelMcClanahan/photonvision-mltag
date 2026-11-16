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
 * Decides how many frames a pipeline keeps in work at the same time.
 *
 * <p>The pipeline has two modes:
 *
 * <ul>
 *   <li>SERIAL: each call completes its own frame. This gives the lowest latency. It is the correct
 *       mode when the pipeline is faster than the camera.
 *   <li>OVERLAP: a call starts the work of its frame and returns the result of an earlier frame.
 *       The work of the frames (model inference on the lanes, then tag decode and pose on the work
 *       thread of the pipeline) runs while the vision thread gets the next camera frames. The
 *       number of frames in work is the depth. This is the correct mode when the work for one frame
 *       takes longer than one camera frame period.
 * </ul>
 *
 * <p>The controller starts in SERIAL mode. It goes to OVERLAP mode when the vision thread has no
 * idle time between calls (the camera is faster than the pipeline) and the inference is the larger
 * part of the work. In OVERLAP mode it increases the depth until the result of the oldest frame is
 * ready when the next camera frame arrives. It goes back to SERIAL mode when the measured camera
 * frame period shows that the SERIAL mode can keep the camera rate. To go back, the pipeline must
 * skip the input of one camera frame for each frame that is in work, because one call can give only
 * one result.
 *
 * <p>All times are in nanoseconds. This class has no connection to frames or models, so a test can
 * give it simulated times.
 */
final class InferenceOverlapController {
    enum Action {
        /** Start the work of the input frame, then return the result of the oldest frame. */
        COMPLETE_OLDEST,
        /** Start the work of the input frame and give no result with this call. */
        DEFER,
        /** Do not use the input frame. Return the result of the oldest frame that is in work. */
        SKIP_INPUT
    }

    /** Idle time below this value means that the vision thread does not wait for the camera. */
    static final long LOW_IDLE_NANOS = 1_500_000L;

    /** Number of calls in sequence without idle time before OVERLAP mode starts. */
    static final int CALLS_TO_START_OVERLAP = 8;

    /** Number of calls in sequence that show that more depth is necessary before it increases. */
    static final int CALLS_TO_ADD_DEPTH = 8;

    /** Number of calls in sequence that show that SERIAL mode is sufficient before OVERLAP stops. */
    static final int CALLS_TO_STOP_OVERLAP = 60;

    /** A time between two frames above this value is a pause, not a frame period. */
    static final long MAX_FRAME_PERIOD_NANOS = 500_000_000L;

    /** The largest number of frames in work. More depth only adds latency. */
    static final int MAX_DEPTH = 4;

    /**
     * Default reserve: 0.3 of a camera frame period. With this reserve the result of a frame is ready
     * before the vision thread needs it also when the work time changes from frame to frame, so the
     * results come at the regular rate of the camera. A smaller reserve can decrease the latency by
     * one camera frame period, but then a part of the results is late and the time between results is
     * irregular.
     */
    static final double DEFAULT_RESERVE = 0.3;

    private final int lanes;

    /**
     * The part of one camera frame period that must stay free between the end of the work for a frame
     * and the time when its result is necessary. A negative value permits a result that is late by
     * that part of a period (the vision thread then waits for it).
     */
    private final double reserve;

    private boolean overlap = false;
    private boolean leavingOverlap = false;
    private int depth = 1;

    private long lastCallEndNanos = 0;
    private long lastFrameTimestampNanos = 0;
    private int lowIdleStreak = 0;
    private int serialSufficientStreak = 0;
    private int moreDepthStreak = 0;

    // The last times between two input frames. The camera frame period is the lower quartile of
    // these times: when frames are skipped, a part of the times is two or more periods, and the
    // timestamps of a camera have some jitter.
    private final long[] recentFramePeriods = new long[32];
    private final long[] sortedFramePeriods = new long[32];
    private int recentFramePeriodCount = 0;
    private int recentFramePeriodNext = 0;

    private double postInferenceAverage = 0;
    private int postInferenceSamples = 0;

    /**
     * The camera frame period that was measured in OVERLAP mode when the pipeline kept the camera
     * rate. 0 = not known. In SERIAL mode with no idle time the camera period cannot be measured,
     * because the pipeline then does not see each camera frame.
     */
    private double knownCameraPeriod = 0;

    InferenceOverlapController(int lanes) {
        this(lanes, DEFAULT_RESERVE);
    }

    InferenceOverlapController(int lanes, double reserve) {
        this.lanes = Math.max(1, lanes);
        this.reserve = Math.max(-0.5, Math.min(0.9, reserve));
    }

    boolean isOverlapActive() {
        return overlap;
    }

    /** Returns the number of frames that stay in work between two calls in OVERLAP mode. */
    int getDepth() {
        return overlap ? depth : 0;
    }

    /**
     * Call this at the start of a pipeline call.
     *
     * @param nowNanos The monotonic time now.
     * @param frameTimestampNanos The capture time of the input frame.
     * @param pendingFrames The number of frames that are in work before this call.
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

        double workNanos = inferenceAverageNanos + postInferenceAverage;

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
            double cameraPeriod = estimateCameraPeriod();
            if (knownCameraPeriod > 0 && cameraPeriod > 0 && cameraPeriod < 0.8 * knownCameraPeriod) {
                knownCameraPeriod = 0;
            }

            lowIdleStreak = idleNanos >= 0 && idleNanos < LOW_IDLE_NANOS ? lowIdleStreak + 1 : 0;
            boolean inferenceIsLargerPart =
                    inferenceSamples >= CALLS_TO_START_OVERLAP && inferenceAverageNanos >= 0.5 * workNanos;
            boolean serialIsSufficient = knownCameraPeriod > 0 && workNanos < 0.9 * knownCameraPeriod;
            if (lowIdleStreak >= CALLS_TO_START_OVERLAP && inferenceIsLargerPart && !serialIsSufficient) {
                overlap = true;
                depth = 1;
                serialSufficientStreak = 0;
                moreDepthStreak = 0;
                // The frame period from SERIAL mode is the pipeline period. Measure it again.
                recentFramePeriodCount = 0;
                recentFramePeriodNext = 0;
            }
        } else {
            // The camera frame period is the lower quartile of the last frame times, not their mean:
            // a pipeline that cannot keep the camera rate does not get each camera frame, so the
            // mean time between its frames is longer than the camera frame period. With the mean, an
            // overloaded pipeline would go back to SERIAL mode, which is slower.
            double cameraPeriod = estimateCameraPeriod();
            boolean serialIsSufficient =
                    cameraPeriod > 0 && inferenceSamples > 0 && workNanos < 0.9 * cameraPeriod;
            serialSufficientStreak = serialIsSufficient ? serialSufficientStreak + 1 : 0;
            if (serialSufficientStreak >= CALLS_TO_STOP_OVERLAP) {
                knownCameraPeriod = cameraPeriod;
                serialSufficientStreak = 0;
                if (pendingFrames > 0) {
                    leavingOverlap = true;
                    return Action.SKIP_INPUT;
                }
                overlap = false;
                lowIdleStreak = 0;
            } else {
                updateDepth(workNanos);
            }
        }

        if (overlap && pendingFrames < depth) {
            return Action.DEFER;
        }
        return Action.COMPLETE_OLDEST;
    }

    /**
     * Increases the depth when the work for one frame is longer than the time that the depth gives.
     * With a depth of d, the result of a frame is necessary d camera frame periods after its start.
     */
    private void updateDepth(double workNanos) {
        double cameraPeriod = estimateCameraPeriod();
        if (cameraPeriod <= 0 || depth >= MAX_DEPTH) {
            moreDepthStreak = 0;
            return;
        }

        boolean moreDepthNecessary = workNanos > (depth - reserve) * cameraPeriod;
        moreDepthStreak = moreDepthNecessary ? moreDepthStreak + 1 : 0;
        if (moreDepthStreak >= CALLS_TO_ADD_DEPTH) {
            depth++;
            moreDepthStreak = 0;
        }
    }

    /**
     * Call this at the end of a pipeline call.
     *
     * @param nowNanos The monotonic time now.
     * @param postInferenceNanos The time of the work after the inference (tag decode and pose) of the
     *     frame that this call returned, or a negative value if this call returned no result.
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

    /** Call this when the frames in work were discarded, for example after new settings. */
    void reset() {
        overlap = false;
        leavingOverlap = false;
        depth = 1;
        lastCallEndNanos = 0;
        lastFrameTimestampNanos = 0;
        lowIdleStreak = 0;
        serialSufficientStreak = 0;
        moreDepthStreak = 0;
        recentFramePeriodCount = 0;
        recentFramePeriodNext = 0;
        knownCameraPeriod = 0;
    }

    /**
     * Returns the camera frame period: the lower quartile of the last times between two input frames.
     * Returns 0 if there are fewer than 8 samples.
     */
    private double estimateCameraPeriod() {
        if (recentFramePeriodCount < 8) {
            return 0;
        }
        System.arraycopy(recentFramePeriods, 0, sortedFramePeriods, 0, recentFramePeriodCount);
        java.util.Arrays.sort(sortedFramePeriods, 0, recentFramePeriodCount);
        return sortedFramePeriods[recentFramePeriodCount / 4];
    }

    private void updateFramePeriod(long frameTimestampNanos) {
        if (lastFrameTimestampNanos > 0 && frameTimestampNanos > lastFrameTimestampNanos) {
            long period = frameTimestampNanos - lastFrameTimestampNanos;
            if (period < MAX_FRAME_PERIOD_NANOS) {
                recentFramePeriods[recentFramePeriodNext] = period;
                recentFramePeriodNext = (recentFramePeriodNext + 1) % recentFramePeriods.length;
                recentFramePeriodCount = Math.min(recentFramePeriods.length, recentFramePeriodCount + 1);
            } else {
                // A pause. The idle time before this call is not a sign of a fast pipeline.
                lowIdleStreak = 0;
            }
        }
        lastFrameTimestampNanos = frameTimestampNanos;
    }
}
