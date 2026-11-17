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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

public class Letterbox {
    private static final Logger logger = new Logger(Letterbox.class, LogGroup.General);
    double dx;
    double dy;
    double scale;

    public Letterbox(double dx, double dy, double scale) {
        this.dx = dx;
        this.dy = dy;
        this.scale = scale;
    }

    /**
     * Resize the frame to the new shape and "letterbox" it.
     *
     * <p>Letterboxing is the process of resizing an image to a new shape while maintaining the aspect
     * ratio of the original image. The new image is padded with a color to fill the remaining space.
     *
     * @param frame
     * @param letterboxed
     * @param newShape
     * @param color
     * @return
     */
    public static Letterbox letterbox(Mat frame, Mat letterboxed, Size newShape, Scalar color) {
        Mat resized = new Mat();
        try {
            return letterbox(frame, resized, letterboxed, newShape, color);
        } finally {
            resized.release();
        }
    }

    /**
     * Resize the frame to the new shape and "letterbox" it. This variant uses a scratch Mat from the
     * caller, so a caller that keeps both Mats does not allocate memory for each frame. The pixel
     * values are the same as the values from {@link #letterbox(Mat, Mat, Size, Scalar)}.
     *
     * @param frame The input image. It can have 1 or 3 channels.
     * @param resizedScratch Scratch Mat for the resized image. It must not be {@code letterboxed}.
     * @param letterboxed The output image. It has the channel count of {@code frame}.
     * @param newShape The output size.
     * @param color The padding color.
     * @return The transform from letterboxed coordinates to frame coordinates.
     */
    public static Letterbox letterbox(
            Mat frame, Mat resizedScratch, Mat letterboxed, Size newShape, Scalar color) {
        // from https://github.com/ultralytics/yolov5/issues/8427#issuecomment-1172469631
        var frameSize = frame.size();
        var r = Math.min(newShape.height / frameSize.height, newShape.width / frameSize.width);

        var newUnpad = new Size(Math.round(frameSize.width * r), Math.round(frameSize.height * r));

        Mat unpadded;
        if (!(frameSize.equals(newUnpad))) {
            Imgproc.resize(frame, resizedScratch, newUnpad, Imgproc.INTER_LINEAR);
            unpadded = resizedScratch;
        } else {
            unpadded = frame;
        }

        var dw = newShape.width - newUnpad.width;
        var dh = newShape.height - newUnpad.height;

        dw /= 2;
        dh /= 2;

        int top = (int) (Math.round(dh - 0.1f));
        int bottom = (int) (Math.round(dh + 0.1f));
        int left = (int) (Math.round(dw - 0.1f));
        int right = (int) (Math.round(dw + 0.1f));
        Core.copyMakeBorder(
                unpadded, letterboxed, top, bottom, left, right, Core.BORDER_CONSTANT, color);

        return new Letterbox(dw, dh, r);
    }

    /**
     * Scratch images for {@link #prepareModelInput}. One detector keeps one of these, so the model
     * input does not allocate memory for each frame. One thread must use it at a time.
     */
    public static final class ModelInputScratch {
        final Mat grayAsBgr = new Mat();
        final Mat resized = new Mat();
        final Mat letterboxed = new Mat();
        final Mat letterboxedBgr = new Mat();
        private boolean lastWasMonochromeFastPath = false;

        /**
         * Returns true if the last call used the 1-channel fast path. The simulation and the benchmark
         * print it, so a measurement says which path it measured.
         */
        public boolean lastWasMonochromeFastPath() {
            return lastWasMonochromeFastPath;
        }

        public void release() {
            grayAsBgr.release();
            resized.release();
            letterboxed.release();
            letterboxedBgr.release();
        }
    }

    /**
     * The model input and the transform back to frame coordinates.
     *
     * @param scale The transform from model input coordinates to frame coordinates.
     * @param image The 3-channel model input. It is a Mat of the scratch, so it is valid until the
     *     next call with the same scratch.
     */
    public record ModelInput(Letterbox scale, Mat image) {}

    /**
     * Makes the 3-channel model input from a camera frame. The bytes are always equal to the base
     * operation, which is a letterbox of a 3-channel image.
     *
     * <p>A monochrome camera gives a 1-channel frame. A resize of 1 channel followed by a copy to 3
     * channels has one third of the resize work. But an OpenCV build can use different resize code
     * for 1 and 3 channels (for example a HAL on ARM), and then the bytes can differ. So the fast
     * path is used only when a self-check on this OpenCV build proved equal bytes for this frame
     * size. The fast path pads the 1-channel image with the first color value only, so it is used
     * only for a gray pad color (three equal values). Otherwise the frame is copied to 3 channels
     * first, which is the base operation.
     *
     * @param in The camera frame with 1 or 3 channels.
     * @param s The scratch images of the caller.
     * @param inputSize The model input size.
     * @param color The padding color.
     * @return The model input.
     */
    public static ModelInput prepareModelInput(
            Mat in, ModelInputScratch s, Size inputSize, Scalar color) {
        boolean useFastPath =
                in.channels() == 1 && monochromeResizeIsExact(in.cols(), in.rows(), inputSize, color);
        return prepareModelInput(in, s, inputSize, color, useFastPath);
    }

    /**
     * The same as {@link #prepareModelInput(Mat, ModelInputScratch, Size, Scalar)}, but the caller
     * selects the path for a 1-channel frame. Only tests use it, so they can test both paths on one
     * host.
     */
    static ModelInput prepareModelInput(
            Mat in, ModelInputScratch s, Size inputSize, Scalar color, boolean useFastPath) {
        if (in.channels() != 1) {
            s.lastWasMonochromeFastPath = false;
            Letterbox scale = letterbox(in, s.resized, s.letterboxed, inputSize, color);
            return new ModelInput(scale, s.letterboxed);
        }
        if (useFastPath) {
            s.lastWasMonochromeFastPath = true;
            Letterbox scale = letterbox(in, s.resized, s.letterboxed, inputSize, color);
            Imgproc.cvtColor(s.letterboxed, s.letterboxedBgr, Imgproc.COLOR_GRAY2BGR);
            return new ModelInput(scale, s.letterboxedBgr);
        }
        // The base operation: three equal channels, then the 3-channel letterbox.
        s.lastWasMonochromeFastPath = false;
        Imgproc.cvtColor(in, s.grayAsBgr, Imgproc.COLOR_GRAY2BGR);
        Letterbox scale = letterbox(s.grayAsBgr, s.resized, s.letterboxed, inputSize, color);
        return new ModelInput(scale, s.letterboxed);
    }

    /**
     * The key of the self-check cache. The Size and Scalar classes are mutable, so the key has plain
     * values. The probe pads with the pad color, so the result is valid only for that color.
     */
    private record ProbeKey(
            int width,
            int height,
            int inputWidth,
            int inputHeight,
            double pad0,
            double pad1,
            double pad2) {}

    private static final ConcurrentHashMap<ProbeKey, Boolean> probeResults =
            new ConcurrentHashMap<>();

    /** Number of self-check runs. Tests read it to prove the cache. */
    static final AtomicInteger probeRuns = new AtomicInteger();

    /** The fast path for a 1-channel frame: a gray image in, the 3-channel model input out. */
    @FunctionalInterface
    interface MonoFastPath {
        void apply(Mat gray, Mat outBgr);
    }

    /**
     * Returns true if the 1-channel fast path gives the same bytes as the base operation for this
     * frame size on this OpenCV build. The self-check runs one time for each size.
     * PV_MLTAG_MONO_MODEL_INPUT=exact always selects the base operation. There is no value that
     * forces the fast path, because that could change the model input. A pad color with unequal
     * channels always gives false: the 1-channel pad has only the first value.
     */
    static boolean monochromeResizeIsExact(int width, int height, Size inputSize, Scalar color) {
        return monochromeResizeIsExact(
                width,
                height,
                inputSize,
                color,
                (gray, outBgr) -> {
                    Mat resized = new Mat();
                    Mat letterboxed = new Mat();
                    try {
                        letterbox(gray, resized, letterboxed, inputSize, color);
                        Imgproc.cvtColor(letterboxed, outBgr, Imgproc.COLOR_GRAY2BGR);
                    } finally {
                        resized.release();
                        letterboxed.release();
                    }
                });
    }

    /**
     * The same as {@link #monochromeResizeIsExact(int, int, Size, Scalar)}, but the caller gives the
     * fast path that the self-check tests. Only tests give a different fast path, so they can put a
     * false result in the cache and prove that the public method then uses the base operation.
     */
    static boolean monochromeResizeIsExact(
            int width, int height, Size inputSize, Scalar color, MonoFastPath fastPath) {
        if (exactModeIsSet()) {
            return false;
        }
        if (!padColorIsGray(color)) {
            // Not cached: the answer does not depend on the OpenCV build.
            return false;
        }
        if (width <= 0 || height <= 0) {
            // An empty frame has no pixels, so both paths give the same result.
            return true;
        }
        var key =
                new ProbeKey(
                        width,
                        height,
                        (int) Math.round(inputSize.width),
                        (int) Math.round(inputSize.height),
                        color.val[0],
                        color.val[1],
                        color.val[2]);
        return probeResults.computeIfAbsent(
                key, k -> probeMonochromeResize(width, height, inputSize, color, fastPath));
    }

    /** True if PV_MLTAG_MONO_MODEL_INPUT=exact is set. Tests read it to know the expected result. */
    static boolean exactModeIsSet() {
        String mode = EnvUtils.string("PV_MLTAG_MONO_MODEL_INPUT", "auto");
        return "exact".equalsIgnoreCase(mode.trim());
    }

    /**
     * The fast path pads the 1-channel image with color.val[0] and copies it to three channels. That
     * is equal to the 3-channel pad only when the three values are equal.
     */
    private static boolean padColorIsGray(Scalar color) {
        return color.val[0] == color.val[1] && color.val[0] == color.val[2];
    }

    /**
     * Compares the fast path with the base operation on two probe pictures of the frame size. Random
     * noise shows each difference of a resize coefficient. The blurred picture has smooth areas,
     * which are like a camera picture.
     */
    static boolean probeMonochromeResize(
            int width, int height, Size inputSize, Scalar color, MonoFastPath fastPath) {
        probeRuns.incrementAndGet();
        Mat noise = new Mat(height, width, CvType.CV_8UC1);
        Mat smooth = new Mat();
        Mat expected = new Mat();
        Mat actual = new Mat();
        Mat bgr = new Mat();
        Mat difference = new Mat();
        boolean exact = true;
        int differentBytes = 0;
        try {
            byte[] data = new byte[width * height];
            new Random(0x5EED1234L).nextBytes(data);
            noise.put(0, 0, data);
            Imgproc.GaussianBlur(noise, smooth, new Size(9, 9), 0);

            for (Mat probe : new Mat[] {noise, smooth}) {
                Imgproc.cvtColor(probe, bgr, Imgproc.COLOR_GRAY2BGR);
                letterbox(bgr, expected, inputSize, color);
                fastPath.apply(probe, actual);
                if (!expected.size().equals(actual.size()) || expected.type() != actual.type()) {
                    exact = false;
                    differentBytes = -1;
                    break;
                }
                if (Core.norm(expected, actual, Core.NORM_INF) != 0) {
                    exact = false;
                    Core.absdiff(expected, actual, difference);
                    // reshape makes a new Mat header that holds the buffer, so release it too.
                    Mat flat = difference.reshape(1);
                    try {
                        differentBytes += Core.countNonZero(flat);
                    } finally {
                        flat.release();
                    }
                }
            }
        } finally {
            for (Mat mat : new Mat[] {noise, smooth, expected, actual, bgr, difference}) {
                mat.release();
            }
        }

        String sizes =
                width
                        + "x"
                        + height
                        + " to "
                        + (int) Math.round(inputSize.width)
                        + "x"
                        + (int) Math.round(inputSize.height);
        if (exact) {
            logger.info(
                    "Monochrome model input "
                            + sizes
                            + ": the 1-channel resize gives the same bytes as the base operation."
                            + " The fast path is used.");
        } else {
            logger.warn(
                    "Monochrome model input "
                            + sizes
                            + ": the 1-channel resize does not give the same bytes as the base operation ("
                            + (differentBytes < 0
                                    ? "different size or type"
                                    : differentBytes + " different bytes")
                            + "). The 3-channel base operation is used, and it costs more time.");
        }
        return exact;
    }

    /**
     * Resizes the detections to the original frame size.
     *
     * @param unscaled The detections to resize
     * @return The resized detections
     */
    public List<NeuralNetworkPipeResult> resizeDetections(List<NeuralNetworkPipeResult> unscaled) {
        var ret = new ArrayList<NeuralNetworkPipeResult>(unscaled.size());

        for (var t : unscaled) {
            var scale = 1.0 / this.scale;
            var boundingBox = t.bbox();

            double cx = (boundingBox.center.x - this.dx) * scale;
            double cy = (boundingBox.center.y - this.dy) * scale;
            double width = boundingBox.size.width * scale;
            double height = boundingBox.size.height * scale;

            Point center = new Point(cx, cy);
            Size size = new Size(width, height);

            // angle is unchanged from letterbox transformation

            ret.add(
                    new NeuralNetworkPipeResult(
                            new RotatedRect(center, size, boundingBox.angle), t.classIdx(), t.confidence()));
        }
        return ret;
    }
}
