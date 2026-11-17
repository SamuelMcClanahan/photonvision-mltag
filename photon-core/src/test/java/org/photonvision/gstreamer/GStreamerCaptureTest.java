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

package org.photonvision.gstreamer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.LoadJNI;
import org.photonvision.vision.opencv.CVMat;

/**
 * Tests for the native GStreamer capture library. The tests run only on a machine that has the
 * library (set PV_GSTREAMER_CAPTURE_JNI_LIB to its path). They use software sources, so a camera is
 * not necessary.
 */
public class GStreamerCaptureTest {
    private static final String APPSINK =
            "appsink name=pvappsink emit-signals=false max-buffers=1 drop=true sync=false";

    @TempDir static Path tempDir;

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
    }

    private static void assumeCapability(long capability) {
        assumeTrue(GStreamerCapture.isAvailable(), "Native GStreamer capture library is not loaded");
        assumeTrue(
                GStreamerCaptureJNI.hasCapability(capability),
                "Native GStreamer capture library does not have capability " + capability);
    }

    private static void assumeElement(String element) {
        assumeTrue(
                GStreamerCaptureJNI.hasElement(element), "GStreamer element is not installed: " + element);
    }

    /** A deterministic test picture that has edges, gradients and noise. */
    private static Mat makeTestImage(int width, int height, boolean color) {
        Mat image = new Mat(height, width, color ? CvType.CV_8UC3 : CvType.CV_8UC1);
        byte[] row = new byte[width * (color ? 3 : 1)];
        Random random = new Random(1234);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int base = (x * 255 / width + y * 255 / height) / 2 + random.nextInt(12);
                if (color) {
                    row[x * 3] = (byte) Math.min(255, base);
                    row[x * 3 + 1] = (byte) Math.min(255, 255 - base / 2);
                    row[x * 3 + 2] = (byte) Math.min(255, (x ^ y) & 0xff);
                } else {
                    row[x] = (byte) Math.min(255, base);
                }
            }
            image.put(y, 0, row);
        }
        Scalar white = color ? new Scalar(255, 255, 255) : new Scalar(255);
        Scalar black = color ? new Scalar(0, 0, 0) : new Scalar(0);
        Imgproc.rectangle(image, new Point(40, 40), new Point(200, 200), white, -1);
        Imgproc.rectangle(image, new Point(80, 80), new Point(160, 160), black, -1);
        Imgproc.circle(image, new Point(width - 120, height - 110), 60, black, 5);
        return image;
    }

    private static Path writeJpeg(String name, Mat image, int... params) {
        Path path = tempDir.resolve(name);
        assertTrue(Imgcodecs.imwrite(path.toString(), image, new MatOfInt(params)));
        return path;
    }

    private static String jpegFilePipeline(Path file) {
        return "multifilesrc location=\""
                + file
                + "\" loop=true caps=image/jpeg,framerate=30/1 ! "
                + APPSINK;
    }

    private static Mat grabCopy(GStreamerCapture capture, int expectedChannels) {
        GStreamerFrame frame = null;
        for (int i = 0; i < 20 && frame == null; i++) {
            frame = capture.grab(500);
        }
        assertNotNull(frame, "The capture gave no frame");
        assertEquals(expectedChannels, frame.channels());
        CVMat mat = frame.toCVMat();
        try {
            assertEquals(frame.width(), mat.getMat().cols());
            assertEquals(frame.height(), mat.getMat().rows());
            assertEquals(expectedChannels, mat.getMat().channels());
            return mat.getMat().clone();
        } finally {
            mat.release();
        }
    }

    private static void assertSamePixels(Mat expected, Mat actual, String message) {
        assertEquals(expected.size(), actual.size(), message + ": size");
        assertEquals(expected.type(), actual.type(), message + ": type");
        Mat difference = new Mat();
        Core.absdiff(expected, actual, difference);
        Mat flat = difference.reshape(1);
        assertEquals(0, Core.countNonZero(flat), message + ": pixels are different");
        difference.release();
    }

    @Test
    public void colorJpegDecodesToExactLuma() {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE);
        assumeElement("multifilesrc");

        Mat source = makeTestImage(644, 482, true);
        Path file = writeJpeg("color420.jpg", source, Imgcodecs.IMWRITE_JPEG_QUALITY, 90);
        Mat expected = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_GRAYSCALE);

        try (GStreamerCapture capture =
                GStreamerCapture.open(jpegFilePipeline(file), GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            // The first frame is decoded in the caller. The next frames come from the worker.
            for (int i = 0; i < 4; i++) {
                Mat actual = grabCopy(capture, 1);
                assertSamePixels(expected, actual, "gray decode of a color JPEG, frame " + i);
                actual.release();
            }
        }
    }

    @Test
    public void chroma422JpegDecodesToExactLuma() {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE);
        assumeElement("multifilesrc");

        Mat source = makeTestImage(1600, 1304, true);
        Path file =
                writeJpeg(
                        "color422.jpg",
                        source,
                        Imgcodecs.IMWRITE_JPEG_QUALITY,
                        85,
                        Imgcodecs.IMWRITE_JPEG_SAMPLING_FACTOR,
                        Imgcodecs.IMWRITE_JPEG_SAMPLING_FACTOR_422);
        Mat expected = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_GRAYSCALE);

        try (GStreamerCapture capture =
                GStreamerCapture.open(jpegFilePipeline(file), GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            Mat actual = grabCopy(capture, 1);
            assertSamePixels(expected, actual, "gray decode of a 4:2:2 JPEG");
            actual.release();
        }
    }

    @Test
    public void colorJpegDecodesToExactBgr() {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE);
        assumeElement("multifilesrc");

        Mat source = makeTestImage(640, 480, true);
        Path file = writeJpeg("colorbgr.jpg", source, Imgcodecs.IMWRITE_JPEG_QUALITY, 90);
        Mat expected = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_COLOR);

        try (GStreamerCapture capture =
                GStreamerCapture.open(jpegFilePipeline(file), GStreamerCaptureJNI.OUTPUT_BGR)) {
            assertNotNull(capture);
            for (int i = 0; i < 3; i++) {
                Mat actual = grabCopy(capture, 3);
                assertSamePixels(expected, actual, "BGR decode, frame " + i);
                actual.release();
            }
        }
    }

    @Test
    public void grayscaleJpegGivesOneChannelInEachMode() {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE);
        assumeElement("multifilesrc");

        Mat source = makeTestImage(640, 480, false);
        Path file = writeJpeg("gray.jpg", source, Imgcodecs.IMWRITE_JPEG_QUALITY, 92);
        Mat expected = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_GRAYSCALE);

        for (int mode :
                new int[] {
                    GStreamerCaptureJNI.OUTPUT_AUTO,
                    GStreamerCaptureJNI.OUTPUT_GRAY,
                    GStreamerCaptureJNI.OUTPUT_BGR
                }) {
            try (GStreamerCapture capture = GStreamerCapture.open(jpegFilePipeline(file), mode)) {
                assertNotNull(capture);
                Mat actual = grabCopy(capture, 1);
                assertSamePixels(expected, actual, "1-component JPEG, mode " + mode);
                actual.release();
            }
        }
    }

    private static double meanSquaredError(Mat reference, Mat image) {
        Mat difference = new Mat();
        Core.absdiff(reference, image, difference);
        difference.convertTo(difference, CvType.CV_32F);
        Core.multiply(difference, difference, difference);
        double mse = Core.mean(difference).val[0];
        difference.release();
        return mse;
    }

    /**
     * Compares the 1-channel image of the direct capture with the image of the legacy pipeline
     * (jpegdec, videoconvert to BGR, then BGR to gray). Both come from the same JPEG of a monochrome
     * camera picture. The direct image must be as near or nearer to the camera picture.
     */
    @Test
    public void directGrayImageIsNotLessAccurateThanLegacyImage() {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE);
        assumeElement("multifilesrc");
        assumeElement("jpegparse");
        assumeElement("jpegdec");

        Mat original = makeTestImage(1600, 1304, false);
        Mat monochromeBgr = new Mat();
        Imgproc.cvtColor(original, monochromeBgr, Imgproc.COLOR_GRAY2BGR);
        // A monochrome UVC camera sends a YCbCr 4:2:2 JPEG with neutral chroma.
        Path file =
                writeJpeg(
                        "mono422.jpg",
                        monochromeBgr,
                        Imgcodecs.IMWRITE_JPEG_QUALITY,
                        90,
                        Imgcodecs.IMWRITE_JPEG_SAMPLING_FACTOR,
                        Imgcodecs.IMWRITE_JPEG_SAMPLING_FACTOR_422);

        Mat direct;
        try (GStreamerCapture capture =
                GStreamerCapture.open(jpegFilePipeline(file), GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            direct = grabCopy(capture, 1);
        }

        Mat legacy = new Mat();
        String legacyPipeline =
                "multifilesrc location=\""
                        + file
                        + "\" loop=true caps=image/jpeg,framerate=30/1 ! jpegparse ! jpegdec ! videoconvert"
                        + " ! video/x-raw,format=BGR ! "
                        + APPSINK;
        try (GStreamerCapture capture = GStreamerCapture.open(legacyPipeline)) {
            assertNotNull(capture);
            Mat legacyBgr = grabCopy(capture, 3);
            Imgproc.cvtColor(legacyBgr, legacy, Imgproc.COLOR_BGR2GRAY);
            legacyBgr.release();
        }

        double directError = meanSquaredError(original, direct);
        double legacyError = meanSquaredError(original, legacy);
        Mat difference = new Mat();
        Core.absdiff(direct, legacy, difference);
        double largestDifference = Core.minMaxLoc(difference).maxVal;
        System.out.println(
                "[capture] error to the camera picture (mean squared): direct="
                        + directError
                        + " legacy="
                        + legacyError
                        + ", largest direct-to-legacy pixel difference="
                        + largestDifference);
        assertTrue(
                directError <= legacyError + 1e-9,
                "The direct image must not have a larger error: " + directError + " > " + legacyError);
        // The two images are the same picture: the difference is only decoder rounding.
        assertTrue(largestDifference <= 6, "Unexpected difference: " + largestDifference);
    }

    /** Removes the Huffman table segments. Many UVC cameras send Motion-JPEG without them. */
    private static byte[] stripHuffmanTables(byte[] jpeg) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        int i = 2;
        out.write(jpeg, 0, 2);
        while (i + 4 <= jpeg.length) {
            int marker = jpeg[i + 1] & 0xff;
            assertEquals(0xff, jpeg[i] & 0xff, "Expected a JPEG marker");
            int length = ((jpeg[i + 2] & 0xff) << 8) | (jpeg[i + 3] & 0xff);
            if (marker == 0xda) {
                // Start of scan: the entropy-coded data follows. Copy all remaining bytes.
                out.write(jpeg, i, jpeg.length - i);
                break;
            }
            if (marker != 0xc4) {
                out.write(jpeg, i, length + 2);
            }
            i += length + 2;
        }
        return out.toByteArray();
    }

    @Test
    public void jpegWithoutHuffmanTablesDecodesToTheSamePixels() throws IOException {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE);
        assumeElement("multifilesrc");

        Mat source = makeTestImage(640, 480, true);
        Path file = writeJpeg("withtables.jpg", source, Imgcodecs.IMWRITE_JPEG_QUALITY, 88);
        Mat expectedGray = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_GRAYSCALE);
        Mat expectedBgr = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_COLOR);

        byte[] original = Files.readAllBytes(file);
        byte[] stripped = stripHuffmanTables(original);
        assertTrue(stripped.length < original.length, "The test JPEG must contain Huffman tables");
        Path strippedFile = tempDir.resolve("notables.jpg");
        Files.write(strippedFile, stripped);

        try (GStreamerCapture capture =
                GStreamerCapture.open(jpegFilePipeline(strippedFile), GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            Mat actual = grabCopy(capture, 1);
            assertSamePixels(expectedGray, actual, "gray decode without Huffman tables");
            actual.release();
        }
        try (GStreamerCapture capture =
                GStreamerCapture.open(jpegFilePipeline(strippedFile), GStreamerCaptureJNI.OUTPUT_BGR)) {
            assertNotNull(capture);
            Mat actual = grabCopy(capture, 3);
            assertSamePixels(expectedBgr, actual, "BGR decode without Huffman tables");
            actual.release();
        }
    }

    @Test
    public void damagedJpegIsNotDelivered() throws IOException {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE | GStreamerCaptureJNI.CAP_STATS);
        assumeElement("multifilesrc");

        Mat source = makeTestImage(640, 480, true);
        Path file = writeJpeg("complete.jpg", source, Imgcodecs.IMWRITE_JPEG_QUALITY, 90);
        byte[] original = Files.readAllBytes(file);
        byte[] truncated = new byte[original.length * 6 / 10];
        System.arraycopy(original, 0, truncated, 0, truncated.length);
        Path truncatedFile = tempDir.resolve("truncated.jpg");
        Files.write(truncatedFile, truncated);

        try (GStreamerCapture capture =
                GStreamerCapture.open(jpegFilePipeline(truncatedFile), GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            for (int i = 0; i < 5; i++) {
                assertNull(capture.grab(100), "A truncated JPEG must not give a frame");
            }
            GStreamerCapture.Stats stats = capture.getStats();
            assertNotNull(stats);
            assertEquals(0, stats.framesDelivered());
            assertTrue(stats.corruptFrames() + stats.decodeErrors() > 0);
        }
    }

    /** Returns the index of the last FF D9 (EOI) marker. */
    private static int lastEoiIndex(byte[] jpeg) {
        for (int i = jpeg.length - 2; i >= 0; i--) {
            if ((jpeg[i] & 0xff) == 0xff && (jpeg[i + 1] & 0xff) == 0xd9) {
                return i;
            }
        }
        throw new AssertionError("The JPEG has no EOI marker");
    }

    /** Inserts zero bytes before the EOI marker, as a camera that pads its frames does. */
    private static byte[] padBeforeEoi(byte[] jpeg, int count) {
        int eoi = lastEoiIndex(jpeg);
        byte[] result = new byte[jpeg.length + count];
        System.arraycopy(jpeg, 0, result, 0, eoi);
        System.arraycopy(jpeg, eoi, result, eoi + count, jpeg.length - eoi);
        return result;
    }

    /** Returns the index of the Se byte in the SOS header. */
    private static int sosSeIndex(byte[] jpeg) {
        int i = 2;
        while (i + 4 <= jpeg.length) {
            assertEquals(0xff, jpeg[i] & 0xff, "Expected a JPEG marker");
            int marker = jpeg[i + 1] & 0xff;
            int length = ((jpeg[i + 2] & 0xff) << 8) | (jpeg[i + 3] & 0xff);
            if (marker == 0xda) {
                int components = jpeg[i + 4] & 0xff;
                // Ns, then two bytes for each component, then Ss, then Se.
                return i + 4 + 1 + 2 * components + 1;
            }
            i += length + 2;
        }
        throw new AssertionError("The JPEG has no SOS marker");
    }

    private static Path writeBytes(String name, byte[] bytes) throws IOException {
        Path path = tempDir.resolve(name);
        Files.write(path, bytes);
        return path;
    }

    private static Mat grabFirstFrameCopy(GStreamerCapture capture) {
        // The first grab decodes the first sample in the caller. A rejected sample gives null.
        GStreamerFrame frame = capture.grab(2000);
        assertNotNull(frame, "The first grab must give a frame");
        CVMat mat = frame.toCVMat();
        try {
            return mat.getMat().clone();
        } finally {
            mat.release();
        }
    }

    @Test
    public void jpegWithShortPaddingBeforeEoiIsDeliveredAtOnce() throws IOException {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE | GStreamerCaptureJNI.CAP_STATS);
        assumeElement("multifilesrc");

        Mat source = makeTestImage(640, 480, true);
        Path file = writeJpeg("padsource.jpg", source, Imgcodecs.IMWRITE_JPEG_QUALITY, 90);
        Mat expected = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_GRAYSCALE);
        Path padded = writeBytes("pad12.jpg", padBeforeEoi(Files.readAllBytes(file), 12));

        try (GStreamerCapture capture =
                GStreamerCapture.open(jpegFilePipeline(padded), GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            Mat actual = grabFirstFrameCopy(capture);
            assertSamePixels(expected, actual, "JPEG with 12 padding bytes");
            actual.release();
            for (int i = 0; i < 5; i++) {
                actual = grabCopy(capture, 1);
                assertSamePixels(expected, actual, "JPEG with 12 padding bytes, frame " + i);
                actual.release();
            }
            assertEquals(0, capture.getStats().corruptFrames());
        }
    }

    @Test
    public void intermittentPaddingDoesNotDropFrames() throws IOException {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE | GStreamerCaptureJNI.CAP_STATS);
        assumeElement("multifilesrc");

        Mat sourceA = makeTestImage(320, 240, true);
        Mat sourceB = new Mat();
        Core.flip(sourceA, sourceB, 1);
        Path fileA = writeJpeg("altsourceA.jpg", sourceA, Imgcodecs.IMWRITE_JPEG_QUALITY, 90);
        Path fileB = writeJpeg("altsourceB.jpg", sourceB, Imgcodecs.IMWRITE_JPEG_QUALITY, 90);
        Mat expectedA = Imgcodecs.imread(fileA.toString(), Imgcodecs.IMREAD_GRAYSCALE);
        Mat expectedB = Imgcodecs.imread(fileB.toString(), Imgcodecs.IMREAD_GRAYSCALE);
        // Frame 0 is clean. Frame 1 has 64 padding bytes before EOI.
        writeBytes("alt-0.jpg", Files.readAllBytes(fileA));
        writeBytes("alt-1.jpg", padBeforeEoi(Files.readAllBytes(fileB), 64));
        String pipeline =
                "multifilesrc location=\""
                        + tempDir.resolve("alt-%d.jpg")
                        + "\" start-index=0 stop-index=1 loop=true caps=image/jpeg,framerate=30/1 ! "
                        + APPSINK;

        try (GStreamerCapture capture =
                GStreamerCapture.open(pipeline, GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            int countA = 0;
            int countB = 0;
            int delivered = 0;
            int grabs = 0;
            while (delivered < 400 && grabs < 2000) {
                grabs++;
                GStreamerFrame frame = capture.grab(500);
                if (frame == null) {
                    continue;
                }
                delivered++;
                CVMat mat = frame.toCVMat();
                try {
                    Mat image = mat.getMat();
                    assertEquals(expectedA.size(), image.size());
                    Mat difference = new Mat();
                    Core.absdiff(expectedA, image, difference);
                    boolean isA = Core.countNonZero(difference) == 0;
                    Core.absdiff(expectedB, image, difference);
                    boolean isB = Core.countNonZero(difference) == 0;
                    difference.release();
                    assertTrue(isA || isB, "Frame " + delivered + " is not one of the two pictures");
                    countA += isA ? 1 : 0;
                    countB += isB ? 1 : 0;
                } finally {
                    mat.release();
                }
            }
            assertEquals(400, delivered, "The capture must deliver 400 frames");
            assertTrue(countA > 0, "The clean picture must be delivered");
            assertTrue(countB > 0, "The padded picture must be delivered");
            GStreamerCapture.Stats stats = capture.getStats();
            assertTrue(stats.corruptFrames() <= 40, "Too many frames dropped: " + stats.corruptFrames());
        }
    }

    @Test
    public void jpegWithNonSequentialSosParametersIsDelivered() throws IOException {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE | GStreamerCaptureJNI.CAP_STATS);
        assumeElement("multifilesrc");

        Mat source = makeTestImage(640, 480, true);
        Path file = writeJpeg("sossource.jpg", source, Imgcodecs.IMWRITE_JPEG_QUALITY, 90);
        Mat expected = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_GRAYSCALE);
        byte[] bytes = Files.readAllBytes(file);
        int se = sosSeIndex(bytes);
        assertEquals(0x3f, bytes[se] & 0xff, "Se of a baseline JPEG");
        bytes[se] = 0;
        Path changed = writeBytes("sose0.jpg", bytes);

        try (GStreamerCapture capture =
                GStreamerCapture.open(jpegFilePipeline(changed), GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            Mat actual = grabFirstFrameCopy(capture);
            assertSamePixels(expected, actual, "JPEG with Se = 0");
            actual.release();
            for (int i = 0; i < 5; i++) {
                actual = grabCopy(capture, 1);
                assertSamePixels(expected, actual, "JPEG with Se = 0, frame " + i);
                actual.release();
            }
            assertEquals(0, capture.getStats().corruptFrames());
        }
    }

    @Test
    public void jpegWithoutEoiMarkerIsDelivered() throws IOException {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE | GStreamerCaptureJNI.CAP_STATS);
        assumeElement("multifilesrc");

        Mat source = makeTestImage(640, 480, true);
        Path file = writeJpeg("eofsource.jpg", source, Imgcodecs.IMWRITE_JPEG_QUALITY, 90);
        Mat expected = Imgcodecs.imread(file.toString(), Imgcodecs.IMREAD_GRAYSCALE);
        byte[] bytes = Files.readAllBytes(file);
        assertEquals(bytes.length - 2, lastEoiIndex(bytes));
        Path noEoi = writeBytes("noeoi.jpg", java.util.Arrays.copyOf(bytes, bytes.length - 2));

        try (GStreamerCapture capture =
                GStreamerCapture.open(jpegFilePipeline(noEoi), GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            Mat actual = grabCopy(capture, 1);
            assertSamePixels(expected, actual, "JPEG without EOI");
            actual.release();
            // Frames are rejected only until the tolerance starts. After that no frame is rejected,
            // so this count is the number of samples before the first delivered frame.
            GStreamerCapture.Stats stats = capture.getStats();
            assertTrue(
                    stats.corruptFrames() + stats.decodeErrors() < 100,
                    "Too many samples before the first frame: " + stats.corruptFrames());
            for (int i = 0; i < 5; i++) {
                actual = grabCopy(capture, 1);
                assertSamePixels(expected, actual, "JPEG without EOI, frame " + i);
                actual.release();
            }
        }
    }

    private static boolean wrapBufferEnabled() {
        String value = System.getenv("PV_GSTREAMER_CAPTURE_WRAP_BUFFER");
        if (value == null || value.isBlank()) {
            return false;
        }
        value = value.trim().toLowerCase();
        return !value.equals("0")
                && !value.equals("false")
                && !value.equals("no")
                && !value.equals("off");
    }

    @Test
    public void lumaPlaneFrameIsCopiedAndCanBeWritten() {
        assumeCapability(GStreamerCaptureJNI.CAP_LUMA_FORMATS | GStreamerCaptureJNI.CAP_STATS);
        assumeElement("videotestsrc");

        String pipeline = "videotestsrc ! video/x-raw,format=NV12,width=326,height=242 ! " + APPSINK;
        try (GStreamerCapture capture =
                GStreamerCapture.open(pipeline, GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            GStreamerFrame frame = capture.grab(2000);
            assertNotNull(frame);
            CVMat mat = frame.toCVMat();
            // An in-place write, as RotateImagePipe and the crosshair drawing do.
            mat.getMat().setTo(new Scalar(0));
            mat.release();
            for (int i = 0; i < 5; i++) {
                GStreamerFrame next = capture.grab(2000);
                assertNotNull(next);
                next.toCVMat().release();
            }
            assertTrue(
                    capture.getStats().convertNanosSum() > 0,
                    "A luma plane must be copied out of the GStreamer buffer");
        }
    }

    @Test
    public void convertedBgrFrameKeepsZeroCopyWhenWrapIsOn() {
        assumeCapability(GStreamerCaptureJNI.CAP_STATS);
        assumeElement("videotestsrc");
        assumeElement("videoconvert");

        String pipeline =
                "videotestsrc ! video/x-raw,width=320,height=240 ! videoconvert ! video/x-raw,format=BGR ! "
                        + APPSINK;
        try (GStreamerCapture capture = GStreamerCapture.open(pipeline)) {
            assertNotNull(capture);
            for (int i = 0; i < 5; i++) {
                GStreamerFrame frame = capture.grab(2000);
                assertNotNull(frame);
                assertEquals(3, frame.channels());
                frame.toCVMat().release();
            }
            long convertNanos = capture.getStats().convertNanosSum();
            if (wrapBufferEnabled()) {
                assertEquals(0, convertNanos, "The legacy pipeline output must stay zero-copy");
            } else {
                assertTrue(convertNanos > 0, "Without the wrap option each frame is copied");
            }
        }
    }

    /** Counts the threads of this process that have the policy SCHED_FIFO (1). */
    static int countFifoThreads() throws IOException {
        int count = 0;
        try (var tasks = Files.list(Path.of("/proc/self/task"))) {
            for (Path task : (Iterable<Path>) tasks::iterator) {
                String stat;
                try {
                    stat = Files.readString(task.resolve("stat"));
                } catch (IOException e) {
                    // The thread ended.
                    continue;
                }
                // Field 2 (the name) can contain spaces. The fields after it start at field 3.
                String[] fields = stat.substring(stat.lastIndexOf(')') + 2).trim().split(" ");
                if (fields.length > 38 && fields[41 - 3].equals("1")) {
                    count++;
                }
            }
        }
        return count;
    }

    /**
     * Runs in a separate JVM, because the library reads PV_GSTREAMER_CAPTURE_RT_PRIORITY only once
     * and the other tests must not run at real-time priority. Prints the thread counts.
     */
    public static final class RealtimePriorityProbe {
        public static void main(String[] args) throws Exception {
            run();
            // Library threads that are not daemon threads would keep this JVM alive.
            System.out.flush();
            System.exit(0);
        }

        private static void run() throws Exception {
            LoadJNI.loadLibraries();
            if (!GStreamerCapture.isAvailable()) {
                System.out.println("PV-RESULT unavailable");
                return;
            }
            int during;
            try (GStreamerCapture capture =
                    GStreamerCapture.open(
                            "videotestsrc is-live=true ! jpegenc ! " + APPSINK,
                            GStreamerCaptureJNI.OUTPUT_GRAY)) {
                if (capture == null) {
                    System.out.println("PV-RESULT open-failed");
                    return;
                }
                for (int i = 0; i < 10; i++) {
                    GStreamerFrame frame = capture.grab(2000);
                    if (frame != null) {
                        frame.toCVMat().release();
                    }
                }
                during = countFifoThreads();
            }
            Thread.sleep(200);
            int after = countFifoThreads();

            int legacy;
            String legacyPipeline =
                    "videotestsrc is-live=true ! video/x-raw,width=640,height=480,framerate=30/1"
                            + " ! jpegenc ! queue max-size-buffers=1 leaky=downstream ! jpegparse ! jpegdec"
                            + " ! videoconvert ! video/x-raw,format=BGR ! "
                            + APPSINK;
            try (GStreamerCapture capture = GStreamerCapture.open(legacyPipeline)) {
                if (capture == null) {
                    System.out.println("PV-RESULT open-failed");
                    return;
                }
                for (int i = 0; i < 10; i++) {
                    GStreamerFrame frame = capture.grab(2000);
                    if (frame != null) {
                        frame.toCVMat().release();
                    }
                }
                legacy = countFifoThreads();
            }
            System.out.println("PV-RESULT " + during + " " + after + " " + legacy);
        }
    }

    @Test
    public void realtimePriorityIsRestoredWhenTheStreamingThreadLeaves() throws Exception {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE);
        assumeElement("videotestsrc");
        assumeElement("jpegenc");
        assumeElement("jpegparse");
        assumeElement("jpegdec");
        assumeTrue(Files.isDirectory(Path.of("/proc/self/task")), "Needs Linux /proc");

        String javaBinary = Path.of(System.getProperty("java.home"), "bin", "java").toString();
        ProcessBuilder builder =
                new ProcessBuilder(
                        javaBinary,
                        "-cp",
                        System.getProperty("java.class.path"),
                        RealtimePriorityProbe.class.getName());
        builder.environment().put("PV_GSTREAMER_CAPTURE_RT_PRIORITY", "10");
        builder.redirectErrorStream(true);
        Path log = tempDir.resolve("rt-probe.log");
        builder.redirectOutput(log.toFile());
        Process process = builder.start();
        boolean ended = process.waitFor(120, java.util.concurrent.TimeUnit.SECONDS);
        if (!ended) {
            process.destroyForcibly();
        }
        String output = Files.readString(log);
        assertTrue(ended, "The probe process did not end: " + output);
        String result = null;
        for (String line : output.split("\n")) {
            if (line.startsWith("PV-RESULT ")) {
                result = line.substring("PV-RESULT ".length()).trim();
            }
        }
        assertNotNull(result, "The probe process gave no result: " + output);
        assumeTrue(!result.equals("unavailable"), "The probe process could not load the library");
        String[] counts = result.split(" ");
        assertEquals(3, counts.length, "Probe result: " + output);
        int during = Integer.parseInt(counts[0]);
        int after = Integer.parseInt(counts[1]);
        int legacy = Integer.parseInt(counts[2]);
        System.out.println(
                "[capture] SCHED_FIFO threads: during=" + during + " after=" + after + " legacy=" + legacy);
        assumeTrue(during > 0, "No thread got SCHED_FIFO. The permission is missing.");
        assertEquals(0, after, "No thread may keep SCHED_FIFO after the capture is closed");
        assertEquals(1, legacy, "Only the source thread may have SCHED_FIFO");
    }

    private record RawLayout(String format, int[] planeStrides, int[] planeRows, int lumaStep) {}

    private static int roundUp4(int value) {
        return (value + 3) & ~3;
    }

    /**
     * Writes raw frames in the default GStreamer memory layout and returns the luma image that the
     * capture must give. Padding and chroma bytes have values that are never in the luma image.
     */
    private static Mat writeRawFrames(Path file, RawLayout layout, int width, int height, int frames)
            throws IOException {
        Mat luma = new Mat(height, width, CvType.CV_8UC1);
        byte[] lumaRow = new byte[width];
        ByteArrayOutputStream frame = new ByteArrayOutputStream();
        int lumaOffset = layout.format().equals("uyvy") ? 1 : 0;

        for (int plane = 0; plane < layout.planeStrides().length; plane++) {
            int stride = layout.planeStrides()[plane];
            for (int y = 0; y < layout.planeRows()[plane]; y++) {
                byte[] row = new byte[stride];
                java.util.Arrays.fill(row, (byte) 0xee);
                if (plane == 0) {
                    for (int x = 0; x < width; x++) {
                        // Values 0..199 only, so 0xee (238) proves a wrong stride or offset.
                        byte value = (byte) ((x * 7 + y * 13) % 200);
                        lumaRow[x] = value;
                        row[x * layout.lumaStep() + lumaOffset] = value;
                    }
                    luma.put(y, 0, lumaRow);
                }
                frame.write(row);
            }
        }

        byte[] frameBytes = frame.toByteArray();
        ByteArrayOutputStream all = new ByteArrayOutputStream();
        for (int i = 0; i < frames; i++) {
            all.write(frameBytes);
        }
        Files.write(file, all.toByteArray());
        return luma;
    }

    @Test
    public void rawFormatsGiveTheExactLumaBytes() throws IOException {
        assumeCapability(GStreamerCaptureJNI.CAP_LUMA_FORMATS);
        assumeElement("rawvideoparse");

        // The width is not a multiple of 4 or 16, so strides have padding and SIMD loops have a tail.
        int width = 326;
        int height = 242;
        int chromaStride = roundUp4((width + 1) / 2);
        List<RawLayout> layouts = new ArrayList<>();
        layouts.add(new RawLayout("gray8", new int[] {roundUp4(width)}, new int[] {height}, 1));
        layouts.add(new RawLayout("yuy2", new int[] {roundUp4(width * 2)}, new int[] {height}, 2));
        layouts.add(new RawLayout("uyvy", new int[] {roundUp4(width * 2)}, new int[] {height}, 2));
        layouts.add(
                new RawLayout(
                        "nv12",
                        new int[] {roundUp4(width), roundUp4(width)},
                        new int[] {height, height / 2},
                        1));
        layouts.add(
                new RawLayout(
                        "i420",
                        new int[] {roundUp4(width), chromaStride, chromaStride},
                        new int[] {height, height / 2, height / 2},
                        1));

        for (RawLayout layout : layouts) {
            Path file = tempDir.resolve("raw-" + layout.format() + ".bin");
            Mat expected = writeRawFrames(file, layout, width, height, 6);
            String pipeline =
                    "filesrc location=\""
                            + file
                            + "\" ! rawvideoparse format="
                            + layout.format()
                            + " width="
                            + width
                            + " height="
                            + height
                            + " framerate=30/1 ! "
                            + APPSINK;
            try (GStreamerCapture capture =
                    GStreamerCapture.open(pipeline, GStreamerCaptureJNI.OUTPUT_GRAY)) {
                assertNotNull(capture, "Could not open the " + layout.format() + " pipeline");
                Mat actual = grabCopy(capture, 1);
                assertSamePixels(expected, actual, "luma of raw " + layout.format());
                actual.release();
            }
            expected.release();
        }
    }

    @Test
    public void slowConsumerAlwaysGetsARecentFrame() throws InterruptedException {
        assumeCapability(GStreamerCaptureJNI.CAP_JPEG_DECODE | GStreamerCaptureJNI.CAP_STATS);
        assumeElement("jpegenc");

        String pipeline =
                "videotestsrc is-live=true pattern=ball ! "
                        + "video/x-raw,format=I420,width=640,height=480,framerate=60/1 ! jpegenc ! "
                        + APPSINK;
        try (GStreamerCapture capture =
                GStreamerCapture.open(pipeline, GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);

            long worstAgeNanos = 0;
            long lastAgeNanos = 0;
            for (int i = 0; i < 30; i++) {
                GStreamerFrame frame = capture.grab(1000);
                assertNotNull(frame);
                assertTrue(frame.estimatedAgeNanos() >= 0, "The frame age must be known");
                worstAgeNanos = Math.max(worstAgeNanos, frame.estimatedAgeNanos());
                lastAgeNanos = frame.estimatedAgeNanos();
                frame.toCVMat().release();
                // The consumer is 4 times slower than the 60 FPS source.
                Thread.sleep(66);
            }

            // If frames collected in a queue, the age would increase by about 50 ms for each grab
            // and reach more than 1 second. The newest-frame design keeps it near one frame period.
            assertTrue(
                    worstAgeNanos < 150_000_000L,
                    "Frame age must stay small, but the worst age was " + worstAgeNanos / 1e6 + " ms");
            assertTrue(lastAgeNanos < 150_000_000L);

            GStreamerCapture.Stats stats = capture.getStats();
            assertNotNull(stats);
            assertEquals(30, stats.framesDelivered());
            assertTrue(stats.sourceBuffers() > 60, "The source must run at its own rate");
            // The consumer used 30 of about 120 camera frames. The other frames must be dropped:
            // after the decode (newest policy, the default) or before it (the other policies).
            long unusedCameraFrames = stats.sourceBuffers() - stats.framesDelivered();
            long droppedBeforeDecode = stats.sourceBuffers() - stats.samplesPulled();
            assertTrue(unusedCameraFrames > 60);
            assertTrue(
                    droppedBeforeDecode + stats.framesReplaced() >= unusedCameraFrames - 3,
                    "Camera frames that are not used must be dropped, not collected");
            String policy = System.getenv("PV_GSTREAMER_CAPTURE_DECODE_POLICY");
            if ("jit".equals(policy)) {
                // Just-in-time: after the first frames (the library first measures the consumer
                // period), no frame is decoded that the consumer does not take. The newest policy
                // decodes about 120 frames here.
                assertTrue(
                        stats.framesProduced() <= stats.framesDelivered() + 15,
                        "Decoded " + stats.framesProduced() + " frames for 30 deliveries");
            } else if (policy == null || policy.isBlank() || "newest".equals(policy)) {
                assertTrue(stats.framesReplaced() > 0, "Old decoded frames must be replaced");
            }
            assertEquals(0, stats.captureTimeMissing());
            assertEquals(0, stats.corruptFrames());
        }
    }

    @Test
    public void captureTimeSurvivesParserDecoderAndConverter() {
        assumeCapability(GStreamerCaptureJNI.CAP_STATS);
        assumeElement("jpegenc");
        assumeElement("jpegparse");
        assumeElement("jpegdec");

        // This is the shape of the legacy pipeline. jpegparse can rewrite the buffer timestamps.
        String pipeline =
                "videotestsrc is-live=true ! video/x-raw,format=I420,width=640,height=480,framerate=60/1"
                        + " ! jpegenc ! queue max-size-buffers=1 leaky=downstream ! jpegparse ! jpegdec"
                        + " ! videoconvert ! video/x-raw,format=BGR ! "
                        + APPSINK;
        try (GStreamerCapture capture = GStreamerCapture.open(pipeline)) {
            assertNotNull(capture);
            for (int i = 0; i < 20; i++) {
                GStreamerFrame frame = capture.grab(1000);
                assertNotNull(frame);
                assertEquals(3, frame.channels());
                assertTrue(frame.estimatedAgeNanos() >= 0);
                assertTrue(frame.estimatedAgeNanos() < 150_000_000L);
                frame.toCVMat().release();
            }
            GStreamerCapture.Stats stats = capture.getStats();
            assertNotNull(stats);
            assertEquals(0, stats.captureTimeMissing(), "The capture time must reach the appsink");
        }
    }

    @Test
    public void decoderOutputPlaneIsUsedAsGrayImage() {
        assumeCapability(GStreamerCaptureJNI.CAP_LUMA_FORMATS);
        assumeElement("jpegenc");
        assumeElement("jpegdec");

        // A decoder in the pipeline (software here, hardware on a coprocessor) gives planar YUV.
        String pipeline =
                "videotestsrc is-live=true ! video/x-raw,format=I420,width=640,height=480,framerate=30/1"
                        + " ! jpegenc ! jpegdec ! "
                        + APPSINK;
        try (GStreamerCapture capture =
                GStreamerCapture.open(pipeline, GStreamerCaptureJNI.OUTPUT_GRAY)) {
            assertNotNull(capture);
            Mat frame = grabCopy(capture, 1);
            assertEquals(640, frame.cols());
            assertEquals(480, frame.rows());
            // The SMPTE test pattern has bright and dark bars.
            Core.MinMaxLocResult range = Core.minMaxLoc(frame);
            assertTrue(range.maxVal - range.minVal > 100);
            frame.release();
        }
    }
}
