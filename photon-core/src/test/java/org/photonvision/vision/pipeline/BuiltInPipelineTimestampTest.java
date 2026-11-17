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

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.photonvision.common.LoadJNI;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

/**
 * The driver mode and focus pipelines return a new output Frame. The capture time of their results
 * must still be the capture time of the camera frame.
 */
public class BuiltInPipelineTimestampTest {
    private static final long CAPTURE_NANOS = 123_456_789L;

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
    }

    private static Frame makeFrame() {
        var props = new FrameStaticProperties(320, 240, 70, null);
        return new Frame(
                7,
                new CVMat(new Mat(240, 320, CvType.CV_8UC3, new Scalar(60, 90, 120))),
                new CVMat(),
                FrameThresholdType.NONE,
                CAPTURE_NANOS,
                props);
    }

    @Test
    public void driverModeResultHasTheCaptureTimeOfTheCameraFrame() {
        var pipeline = new DriverModePipeline();
        CVPipelineResult result = pipeline.run(makeFrame(), QuirkyCamera.DefaultCamera);
        assertEquals(CAPTURE_NANOS, result.getImageCaptureTimestampNanos());
        result.release();
    }

    @Test
    public void focusResultHasTheCaptureTimeOfTheCameraFrame() {
        var pipeline = new FocusPipeline();
        CVPipelineResult result = pipeline.run(makeFrame(), QuirkyCamera.DefaultCamera);
        assertEquals(CAPTURE_NANOS, result.getImageCaptureTimestampNanos());
        result.release();
    }
}
