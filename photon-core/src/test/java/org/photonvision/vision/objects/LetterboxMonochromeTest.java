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
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.awt.Color;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.util.ColorHelper;

/**
 * A monochrome camera gives a 1-channel image. The detectors use the fast path (resize 1 channel,
 * then copy the gray value into three channels) only when a self-check proved equal bytes on the
 * OpenCV build that runs. Otherwise they copy to three channels first, which is the base operation.
 * These tests prove both paths against the base operation. The fast path is proven on this host
 * only: an ARM build of OpenCV can use other resize code, and the self-check covers that case.
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
        Letterbox.ModelInputScratch scratch = new Letterbox.ModelInputScratch();
        Letterbox.ModelInput input =
                Letterbox.prepareModelInput(gray, scratch, modelSize, padColor, true);
        Letterbox actualScale = input.scale();
        Mat actual = input.image();
        assertFastPathStructure(scratch, input);

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

        for (Mat mat : new Mat[] {gray, bgr, expected, difference}) {
            mat.release();
        }
        scratch.release();
    }

    /**
     * On this host both paths give the same bytes, so a byte check cannot see which path ran. These
     * checks prove the operations of the fast path by the Mats that it writes.
     */
    private static void assertFastPathStructure(
            Letterbox.ModelInputScratch scratch, Letterbox.ModelInput input) {
        assertTrue(scratch.lastWasMonochromeFastPath());
        assertSame(scratch.letterboxedBgr, input.image());
        assertEquals(1, scratch.letterboxed.channels());
        assertEquals(3, input.image().channels());
    }

    /**
     * Proves the operations of the base path: the frame is copied to 3 channels first, and the
     * 3-channel image is letterboxed.
     */
    private static void assertBasePathStructure(
            Letterbox.ModelInputScratch scratch, Letterbox.ModelInput input, int width, int height) {
        assertFalse(scratch.lastWasMonochromeFastPath());
        assertSame(scratch.letterboxed, input.image());
        assertEquals(new Size(width, height), scratch.grayAsBgr.size());
        assertEquals(3, scratch.grayAsBgr.channels());
        assertEquals(3, scratch.letterboxed.channels());
    }

    private static final int[][] SIZES = {
        {1600, 1304}, // ThriftyCam
        {1280, 800}, // OV9281
        {640, 480},
        {640, 640}, // No resize
        {1304, 1600} // Rotated camera
    };

    private static void assertSameBytes(Mat expected, Mat actual, String message) {
        assertEquals(expected.size(), actual.size(), message);
        assertEquals(expected.type(), actual.type(), message);
        assertEquals(0.0, Core.norm(expected, actual, Core.NORM_INF), message);
    }

    /** The real fast path, the same as the one that the self-check tests in production. */
    private static void realFastPath(Mat gray, Mat outBgr, Size modelSize, Scalar padColor) {
        Mat resized = new Mat();
        Mat letterboxed = new Mat();
        Letterbox.letterbox(gray, resized, letterboxed, modelSize, padColor);
        Imgproc.cvtColor(letterboxed, outBgr, Imgproc.COLOR_GRAY2BGR);
        resized.release();
        letterboxed.release();
    }

    @Test
    public void probeDetectsInexactFastPath() {
        Size modelSize = new Size(640, 640);
        Scalar padColor = ColorHelper.colorToScalar(Color.GRAY);
        // A fast path that is wrong in one byte: the probe must find it.
        boolean brokenResult =
                Letterbox.probeMonochromeResize(
                        1280,
                        800,
                        modelSize,
                        padColor,
                        (gray, out) -> {
                            realFastPath(gray, out, modelSize, padColor);
                            double[] value = out.get(300, 200);
                            value[1] = ((int) value[1] + 1) % 256;
                            out.put(300, 200, value);
                        });
        assertFalse(brokenResult);

        // The probe uses two pictures. A fast path that is wrong only for one of them must also be
        // found, so the probe must check both pictures.
        for (int wrongCall = 0; wrongCall < 2; wrongCall++) {
            final int callToChange = wrongCall;
            int[] calls = {0};
            boolean result =
                    Letterbox.probeMonochromeResize(
                            1280,
                            800,
                            modelSize,
                            padColor,
                            (gray, out) -> {
                                realFastPath(gray, out, modelSize, padColor);
                                if (calls[0]++ == callToChange) {
                                    double[] value = out.get(300, 200);
                                    value[1] = ((int) value[1] + 1) % 256;
                                    out.put(300, 200, value);
                                }
                            });
            assertEquals(2, calls[0], "The probe must test both pictures");
            assertFalse(
                    result,
                    callToChange == 0
                            ? "Wrong only for the noise picture"
                            : "Wrong only for the blurred picture");
        }

        // The real fast path is exact on this host. This is proven on this host only.
        boolean realResult =
                Letterbox.probeMonochromeResize(
                        1280,
                        800,
                        modelSize,
                        padColor,
                        (gray, out) -> realFastPath(gray, out, modelSize, padColor));
        assertTrue(realResult);
    }

    @Test
    public void basePathEqualsBase() {
        Size modelSize = new Size(640, 640);
        Scalar padColor = ColorHelper.colorToScalar(Color.GRAY);
        Letterbox.ModelInputScratch scratch = new Letterbox.ModelInputScratch();
        for (int[] size : SIZES) {
            int width = size[0];
            int height = size[1];
            Mat gray = randomGray(width, height, width * 17L + height);
            Mat bgr = new Mat();
            Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);
            Mat expected = new Mat();
            Letterbox expectedScale = Letterbox.letterbox(bgr, expected, modelSize, padColor);

            // Two calls with the same scratch: the detectors keep the Mats between frames.
            for (int call = 0; call < 2; call++) {
                Letterbox.ModelInput input =
                        Letterbox.prepareModelInput(gray, scratch, modelSize, padColor, false);
                assertBasePathStructure(scratch, input, width, height);
                assertSameBytes(expected, input.image(), "Base path for " + width + "x" + height);
                assertEquals(expectedScale.dx, input.scale().dx);
                assertEquals(expectedScale.dy, input.scale().dy);
                assertEquals(expectedScale.scale, input.scale().scale);
            }
            gray.release();
            bgr.release();
            expected.release();
        }

        // A color frame with different channels in the same scratch.
        Mat gray = randomGray(1280, 800, 99);
        Mat color = new Mat();
        Imgproc.cvtColor(gray, color, Imgproc.COLOR_GRAY2BGR);
        Core.add(color, new Scalar(3, 40, 90), color);
        Mat expected = new Mat();
        Letterbox expectedScale = Letterbox.letterbox(color, expected, modelSize, padColor);
        Letterbox.ModelInput input =
                Letterbox.prepareModelInput(color, scratch, modelSize, padColor, false);
        assertSameBytes(expected, input.image(), "Color frame");
        assertEquals(expectedScale.scale, input.scale().scale);
        input = Letterbox.prepareModelInput(color, scratch, modelSize, padColor);
        assertSameBytes(expected, input.image(), "Color frame, public method");

        gray.release();
        color.release();
        expected.release();
        scratch.release();
    }

    @Test
    public void probeResultIsCached() {
        Size modelSize = new Size(640, 640);
        Scalar padColor = ColorHelper.colorToScalar(Color.GRAY);
        // A size that no other test uses, so the cache has no entry for it.
        int before = Letterbox.probeRuns.get();
        boolean first = Letterbox.monochromeResizeIsExact(1111, 777, modelSize, padColor);
        boolean second = Letterbox.monochromeResizeIsExact(1111, 777, modelSize, padColor);
        assertEquals(first, second);
        if (Letterbox.exactModeIsSet()) {
            // PV_MLTAG_MONO_MODEL_INPUT=exact selects the base operation before the probe.
            assertFalse(first);
            assertEquals(0, Letterbox.probeRuns.get() - before);
        } else {
            assertEquals(1, Letterbox.probeRuns.get() - before);
        }
    }

    /** Asserts the base path and the bytes of the base operation for a 1-channel frame. */
    private static void assertPublicMethodGivesBase(
            Mat gray, Size modelSize, Scalar padColor, String message) {
        Mat bgr = new Mat();
        Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);
        Mat expected = new Mat();
        Letterbox expectedScale = Letterbox.letterbox(bgr, expected, modelSize, padColor);

        Letterbox.ModelInputScratch scratch = new Letterbox.ModelInputScratch();
        Letterbox.ModelInput input = Letterbox.prepareModelInput(gray, scratch, modelSize, padColor);
        assertBasePathStructure(scratch, input, gray.cols(), gray.rows());
        assertSameBytes(expected, input.image(), message);
        assertEquals(expectedScale.dx, input.scale().dx);
        assertEquals(expectedScale.dy, input.scale().dy);
        assertEquals(expectedScale.scale, input.scale().scale);

        bgr.release();
        expected.release();
        scratch.release();
    }

    @Test
    public void publicMethodUsesBasePathAfterFalseSelfCheck() {
        Size modelSize = new Size(640, 640);
        Scalar padColor = ColorHelper.colorToScalar(Color.GRAY);
        // A size that no other test uses, so the cache gets the false result of the broken fast path.
        int width = 1000;
        int height = 700;
        boolean exact =
                Letterbox.monochromeResizeIsExact(
                        width,
                        height,
                        modelSize,
                        padColor,
                        (gray, out) -> {
                            realFastPath(gray, out, modelSize, padColor);
                            double[] value = out.get(300, 200);
                            value[1] = ((int) value[1] + 1) % 256;
                            out.put(300, 200, value);
                        });
        assertFalse(exact);
        int before = Letterbox.probeRuns.get();

        // The public method must use the cached false result and select the base operation.
        Mat gray = randomGray(width, height, 5151);
        assertPublicMethodGivesBase(gray, modelSize, padColor, "After a false self-check");
        assertEquals(before, Letterbox.probeRuns.get());
        gray.release();
    }

    @Test
    public void padColorWithUnequalChannelsUsesBasePath() {
        Size modelSize = new Size(640, 640);
        int width = 1280;
        int height = 800;
        Mat gray = randomGray(width, height, 6262);

        // First a gray pad, so the cache can have a true result for this frame size.
        Scalar grayPad = new Scalar(128, 128, 128);
        Letterbox.ModelInputScratch scratch = new Letterbox.ModelInputScratch();
        Letterbox.prepareModelInput(gray, scratch, modelSize, grayPad);
        scratch.release();

        // The fast path pads 1 channel with the first value only. This pad must use the base path.
        Scalar greenPad = new Scalar(0, 255, 0);
        int before = Letterbox.probeRuns.get();
        assertFalse(Letterbox.monochromeResizeIsExact(width, height, modelSize, greenPad));
        assertPublicMethodGivesBase(gray, modelSize, greenPad, "Pad with unequal channels");
        // The answer for such a pad does not come from a probe.
        assertEquals(before, Letterbox.probeRuns.get());
        gray.release();
    }

    @Test
    public void publicMethodUsesSelfCheckResult() {
        Size modelSize = new Size(640, 640);
        Scalar padColor = ColorHelper.colorToScalar(Color.GRAY);
        // A size that no other test uses. Width and height differ, so an exchange is visible.
        int width = 1234;
        int height = 567;
        boolean exact = Letterbox.monochromeResizeIsExact(width, height, modelSize, padColor);
        int before = Letterbox.probeRuns.get();

        Mat gray = randomGray(width, height, 4242);
        Mat bgr = new Mat();
        Imgproc.cvtColor(gray, bgr, Imgproc.COLOR_GRAY2BGR);
        Mat expected = new Mat();
        Letterbox expectedScale = Letterbox.letterbox(bgr, expected, modelSize, padColor);

        Letterbox.ModelInputScratch scratch = new Letterbox.ModelInputScratch();
        Letterbox.ModelInput input = Letterbox.prepareModelInput(gray, scratch, modelSize, padColor);

        // The public method must select the path from the self-check result for this frame size.
        assertEquals(exact, scratch.lastWasMonochromeFastPath());
        // No new probe: the public method asked for the same width and height, which are cached.
        assertEquals(before, Letterbox.probeRuns.get());
        if (exact) {
            assertFastPathStructure(scratch, input);
        } else {
            assertBasePathStructure(scratch, input, width, height);
        }
        assertSameBytes(expected, input.image(), "Public method for " + width + "x" + height);
        assertEquals(expectedScale.dx, input.scale().dx);
        assertEquals(expectedScale.dy, input.scale().dy);
        assertEquals(expectedScale.scale, input.scale().scale);

        for (Mat mat : new Mat[] {gray, bgr, expected}) {
            mat.release();
        }
        scratch.release();
    }

    @Test
    public void fastPathEqualsBaseOnThisHost() {
        // Proven on this host only. The self-check decides on each other host.
        for (int[] size : SIZES) {
            assertModelInputIsEqual(size[0], size[1]);
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
