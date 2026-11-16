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

package org.photonvision.vision.objects;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.awt.Color;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.util.ColorHelper;

/**
 * A monochrome camera gives a 1-channel image. The detectors resize that image and then copy the
 * gray value into three channels. These tests prove that the model input is then byte-for-byte the
 * same as the model input from a 3-channel image of the same camera frame.
 */
public class LetterboxMonochromeTest {
    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
    }

    private static Mat randomGray(int width, int height, long seed) {
        Mat gray = new Mat(height, width, CvType.CV_8UC1);
        byte[] data = new byte[width * height];
        new Random(seed).nextBytes(data);
        gray.put(0, 0, data);
        // Blur a part of the picture so it has smooth areas and noise areas.
        Mat region = gray.submat(0, height / 2, 0, width);
        Imgproc.GaussianBlur(region, region, new Size(9, 9), 0);
        region.release();
        return gray;
    }

    private static void assertModelInputIsEqual(int width, int height) {
        Size modelSize = new Size(640, 640);
        var padColor = ColorHelper.colorToScalar(Color.GRAY);

        Mat gray = randomGray(width, height, width * 31L + height);
        Mat bgr = new Mat();
        Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);

        // The path for a color image: letterbox the 3-channel image.
        Mat expected = new Mat();
        Letterbox expectedScale = Letterbox.letterbox(bgr, expected, modelSize, padColor);

        // The path for a monochrome image: letterbox 1 channel, then copy it to 3 channels.
        Mat scratch = new Mat();
        Mat grayLetterboxed = new Mat();
        Letterbox actualScale =
                Letterbox.letterbox(gray, scratch, grayLetterboxed, modelSize, padColor);
        Mat actual = new Mat();
        Imgproc.cvtColor(grayLetterboxed, actual, Imgproc.COLOR_GRAY2BGR);

        assertEquals(expected.size(), actual.size());
        assertEquals(expected.type(), actual.type());
        Mat difference = new Mat();
        Core.absdiff(expected, actual, difference);
        assertEquals(
                0,
                Core.countNonZero(difference.reshape(1)),
                "Model input must be equal for " + width + "x" + height);

        assertEquals(expectedScale.dx, actualScale.dx);
        assertEquals(expectedScale.dy, actualScale.dy);
        assertEquals(expectedScale.scale, actualScale.scale);

        for (Mat mat : new Mat[] {gray, bgr, expected, scratch, grayLetterboxed, actual, difference}) {
            mat.release();
        }
    }

    @Test
    public void monochromeModelInputEqualsColorModelInput() {
        assertModelInputIsEqual(1600, 1304); // ThriftyCam
        assertModelInputIsEqual(1280, 800); // OV9281
        assertModelInputIsEqual(640, 480);
        assertModelInputIsEqual(640, 640); // No resize
        assertModelInputIsEqual(1304, 1600); // Rotated camera
    }

    @Test
    public void scratchVariantEqualsOriginalVariant() {
        Size modelSize = new Size(640, 640);
        var padColor = ColorHelper.colorToScalar(Color.GRAY);
        Mat gray = randomGray(1280, 800, 77);
        Mat bgr = new Mat();
        Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);
        // Make the channels different so a channel error is visible.
        Core.add(bgr, new org.opencv.core.Scalar(3, 40, 90), bgr);

        Mat original = new Mat();
        Letterbox.letterbox(bgr, original, modelSize, padColor);

        Mat scratch = new Mat();
        Mat reused = new Mat();
        // Call two times: the Mats keep their memory between frames in the detectors.
        Letterbox.letterbox(bgr, scratch, reused, modelSize, padColor);
        Letterbox.letterbox(bgr, scratch, reused, modelSize, padColor);

        Mat difference = new Mat();
        Core.absdiff(original, reused, difference);
        assertEquals(0, Core.countNonZero(difference.reshape(1)));
    }
}
