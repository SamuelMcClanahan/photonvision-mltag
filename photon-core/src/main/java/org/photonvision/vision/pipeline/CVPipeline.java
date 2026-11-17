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

import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

public abstract class CVPipeline<R extends CVPipelineResult, S extends CVPipelineSettings>
        implements Releasable {
    protected S settings;
    protected FrameStaticProperties frameStaticProperties;
    protected QuirkyCamera cameraQuirks;

    private final FrameThresholdType thresholdType;

    // So releaseable doesn't keep track of if we double-free something. so (ew) remember that here
    protected volatile boolean released = false;

    public CVPipeline(FrameThresholdType thresholdType) {
        this.thresholdType = thresholdType;
    }

    public FrameThresholdType getThresholdType() {
        return thresholdType;
    }

    protected void setPipeParams(
            FrameStaticProperties frameStaticProperties, S settings, QuirkyCamera cameraQuirks) {
        this.settings = settings;
        this.frameStaticProperties = frameStaticProperties;
        this.cameraQuirks = cameraQuirks;

        setPipeParamsImpl();
    }

    protected abstract void setPipeParamsImpl();

    protected abstract R process(Frame frame, S settings);

    public S getSettings() {
        return settings;
    }

    public void setSettings(S s) {
        this.settings = s;
    }

    public R run(Frame frame, QuirkyCamera cameraQuirks) {
        if (released) {
            throw new RuntimeException("Pipeline use-after-free!");
        }
        if (settings == null) {
            throw new RuntimeException("No settings provided for pipeline!");
        }
        setPipeParams(frame.frameStaticProperties, settings, cameraQuirks);

        // if (frame.image.getMat().empty()) {
        //     //noinspection unchecked
        //     return (R) new CVPipelineResult(0, 0, List.of(), frame);
        // }
        R result = process(frame, settings);

        // A pipeline that keeps several frames in work at the same time (see AprilTagPipeline) can
        // give no result for this call. The frame stays with the pipeline, and its result comes
        // with a later call.
        if (result == null) {
            return null;
        }

        // The capture time must always be the capture time of the camera frame that the result is
        // computed from. Some pipelines put a new output Frame into the result, so the result frame
        // is not that camera frame in general.
        result.setImageCaptureTimestampNanos(getResultSourceFrame(result, frame).timestampNanos);

        return result;
    }

    /**
     * Returns the camera frame that a result was computed from. This is the input frame of the call
     * for most pipelines. A pipeline that can return the result of an earlier frame (see
     * AprilTagPipeline) overrides this.
     *
     * @param result The result of a call of {@link #run}.
     * @param inputFrame The input frame of that call.
     */
    public Frame getResultSourceFrame(R result, Frame inputFrame) {
        return inputFrame;
    }

    /**
     * Returns a result of a frame that the pipeline kept in work and that is complete now, or null.
     * VisionRunner calls this when there is no camera frame for a call of {@link #run}, so these
     * results are not lost. Most pipelines keep no frame between calls and return null.
     */
    public R pollCompletedResult() {
        return null;
    }

    /**
     * Release any native memory associated with this pipeline. Called by pipelinemanager at pipeline
     * switch. Stubbed out, but override if needed.
     */
    @Override
    public void release() {
        released = true;
    }
}
