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

package org.photonvision.vision.frame.provider;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;

import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.photonvision.common.LoadJNI;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameDivisor;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipeline.AprilTagPipelineSettings;
import org.photonvision.vision.pipeline.OutputStreamPipeline;
import org.photonvision.vision.pipeline.StreamDemand;

public class CpuImageProcessorTest {
    private static final int WIDTH = 64;
    private static final int HEIGHT = 48;

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
    }

    /** Gives a copy of the same 1-channel picture for each frame. */
    private static class GrayProcessor extends CpuImageProcessor {
        final Mat picture;

        GrayProcessor(Mat picture) {
            this.picture = picture;
        }

        @Override
        CapturedFrame getInputMat() {
            return new CapturedFrame(
                    new CVMat(picture.clone()), new FrameStaticProperties(WIDTH, HEIGHT, 70, null), 1);
        }

        @Override
        protected boolean checkCameraConnected() {
            return true;
        }

        @Override
        public String getName() {
            return "GrayProcessor";
        }

        @Override
        public void release() {
            releaseProcessorResources();
        }
    }

    private static Mat makeGrayPicture() {
        Mat picture = new Mat(HEIGHT, WIDTH, CvType.CV_8UC1);
        byte[] bytes = new byte[WIDTH * HEIGHT];
        new Random(42).nextBytes(bytes);
        picture.put(0, 0, bytes);
        return picture;
    }

    private static void assertSameBytes(Mat expected, Mat actual) {
        assertEquals(expected.channels(), actual.channels(), "channels");
        assertEquals(expected.size(), actual.size(), "size");
        Mat difference = new Mat();
        Core.absdiff(expected, actual, difference);
        assertEquals(0, Core.countNonZero(difference), "The input image was changed");
        difference.release();
    }

    @Test
    public void outputStreamDoesNotDrawOnTheInputImage() {
        Mat picture = makeGrayPicture();
        var processor = new GrayProcessor(picture);
        processor.requestFrameThresholdType(FrameThresholdType.GREYSCALE);
        processor.requestFrameCopies(false, true);

        Frame frame = processor.get();
        assertNotEquals(
                frame.colorImage.getMat().dataAddr(),
                frame.processedImage.getMat().dataAddr(),
                "The processed image must not be the input image when the output stream needs it");

        var settings = new AprilTagPipelineSettings();
        settings.outputShouldShow = true;
        settings.outputShouldDraw = true;
        settings.streamingFrameDivisor = FrameDivisor.HALF;
        // The stream pipeline resizes the processed image, makes it BGR and draws on it in place.
        new OutputStreamPipeline()
                .process(frame, settings, List.of(), List.of(), new StreamDemand(false, true));

        assertEquals(1, frame.colorImage.getMat().channels());
        assertSameBytes(picture, frame.colorImage.getMat());

        frame.release();
        processor.release();
        picture.release();
    }

    @Test
    public void noStreamKeepsTheZeroCopyPath() {
        Mat picture = makeGrayPicture();
        var processor = new GrayProcessor(picture);
        processor.requestFrameThresholdType(FrameThresholdType.GREYSCALE);
        processor.requestFrameCopies(false, false);

        Frame frame = processor.get();
        // No stream draws on the frame, so the processed image is the input image with no copy.
        assertEquals(frame.colorImage.getMat().dataAddr(), frame.processedImage.getMat().dataAddr());
        assertSameBytes(picture, frame.processedImage.getMat());

        frame.release();
        processor.release();
        picture.release();
    }
}
