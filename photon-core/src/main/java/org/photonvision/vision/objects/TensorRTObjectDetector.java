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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.objects;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.tensorrt.TensorRTJNI;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

/** Jetson TensorRT object detector backed by PhotonVision's JNI bridge. */
public class TensorRTObjectDetector implements ObjectDetector {
    private static final Logger logger = new Logger(TensorRTObjectDetector.class, LogGroup.General);
    private static final int GSTREAMER_FRAME_LOG_EVERY =
            EnvUtils.nonNegativeInteger("PV_TENSORRT_GSTREAMER_FRAME_LOG_EVERY", 0);
    private static final boolean DISABLE_GSTREAMER_FRAME_DETECT =
            EnvUtils.flag("PV_TENSORRT_DISABLE_GSTREAMER_FRAME");
    private static final Object NATIVE_STARTUP_LOCK = new Object();
    private static final AtomicInteger NATIVE_STARTUP_SERIAL_REQUESTS =
            new AtomicInteger(EnvUtils.nonNegativeInteger("PV_TENSORRT_STARTUP_SERIAL_REQUESTS", 16));

    private final TensorRTModel model;
    private long nativePtr;
    private long nativeGStreamerFrameCount = 0;

    /** Returns the model in use by this detector. */
    @Override
    public TensorRTModel getModel() {
        return model;
    }

    /**
     * Creates a new TensorRTObjectDetector from the given model.
     *
     * @param model The model to create the detector from.
     * @param inputSize The required image dimensions for the model.
     */
    public TensorRTObjectDetector(TensorRTModel model, Size inputSize) {
        this.model = model;
        startNative(inputSize);
    }

    /** Returns the classes that the detector can detect. */
    @Override
    public List<String> getClasses() {
        return model.properties.labels();
    }

    /**
     * Detects objects in the given input image using TensorRT. The JNI layer performs letterbox
     * preprocessing and returns detections in original image coordinates.
     */
    @Override
    public synchronized List<NeuralNetworkPipeResult> detect(
            Mat in, double nmsThresh, double boxThresh) {
        if (!isValid()) {
            logger.error("TensorRT JNI detector is not initialized! Model: " + model.modelFile.getName());
            return List.of();
        }

        return detectNativeOriginal(in, nmsThresh, boxThresh);
    }

    @Override
    public synchronized List<NeuralNetworkPipeResult> detect(
            CVMat in, double nmsThresh, double boxThresh) {
        if (!isValid()) {
            logger.error("TensorRT JNI detector is not initialized! Model: " + model.modelFile.getName());
            return List.of();
        }

        if (in.getNativeFrameHandle() != 0 && !DISABLE_GSTREAMER_FRAME_DETECT) {
            var nativeResults =
                    detectNativeGStreamerFrame(in.getNativeFrameHandle(), nmsThresh, boxThresh);
            if (nativeResults != null) {
                return nativeResults;
            }
        }

        return detectNativeOriginal(in.getMat(), nmsThresh, boxThresh);
    }

    /** Thread-safe method to release the detector. */
    @Override
    public synchronized void release() {
        if (nativePtr != 0) {
            TensorRTJNI.destroy(nativePtr);
            nativePtr = 0;
        }
    }

    private void startNative(Size inputSize) {
        if (EnvUtils.flag("PV_TENSORRT_DISABLE_JNI")) {
            logger.debug("TensorRT JNI detector disabled by PV_TENSORRT_DISABLE_JNI");
            return;
        }

        if (!TensorRTJNI.isAvailable()) {
            logger.error("TensorRT JNI library is unavailable");
            return;
        }

        try {
            nativePtr =
                    TensorRTJNI.create(
                            model.modelFile.getAbsolutePath(),
                            model.properties.resolutionWidth(),
                            model.properties.resolutionHeight(),
                            model.properties.labels().size());
            if (nativePtr == 0) {
                logger.error("TensorRT JNI returned a null detector for " + model.modelFile.getName());
                return;
            }
            logger.debug(
                    "Created TensorRT JNI detector for model "
                            + model.modelFile.getName()
                            + " inputSize="
                            + inputSize);
        } catch (Throwable e) {
            nativePtr = 0;
            logger.error("Failed to create TensorRT JNI detector", e);
        }
    }

    private List<NeuralNetworkPipeResult> detectNativeOriginal(
            Mat input, double nmsThresh, double boxThresh) {
        try {
            double[] flat =
                    runNativeStartupSerialized(
                            () ->
                                    TensorRTJNI.detectOriginal(
                                            nativePtr, input.getNativeObjAddr(), nmsThresh, boxThresh));
            return flatToResults(flat);
        } catch (Throwable e) {
            logger.error("TensorRT JNI detector request failed; releasing detector", e);
            release();
            return List.of();
        }
    }

    private List<NeuralNetworkPipeResult> detectNativeGStreamerFrame(
            long nativeFrameHandle, double nmsThresh, double boxThresh) {
        try {
            double[] flat =
                    runNativeStartupSerialized(
                            () ->
                                    TensorRTJNI.detectGStreamerFrame(
                                            nativePtr, nativeFrameHandle, nmsThresh, boxThresh));
            nativeGStreamerFrameCount++;
            if (GSTREAMER_FRAME_LOG_EVERY > 0
                    && nativeGStreamerFrameCount % GSTREAMER_FRAME_LOG_EVERY == 0) {
                logger.info(
                        "TensorRT JNI processed " + nativeGStreamerFrameCount + " GStreamer NVMM frames");
            }
            return flatToResults(flat);
        } catch (UnsupportedOperationException e) {
            return null;
        } catch (Throwable e) {
            logger.error("TensorRT JNI GStreamer frame request failed; falling back to Mat path", e);
            return null;
        }
    }

    private static double[] runNativeStartupSerialized(NativeDetectRequest request) {
        if (NATIVE_STARTUP_SERIAL_REQUESTS.get() <= 0) {
            return request.run();
        }

        synchronized (NATIVE_STARTUP_LOCK) {
            NATIVE_STARTUP_SERIAL_REQUESTS.updateAndGet(value -> Math.max(0, value - 1));
            return request.run();
        }
    }

    @FunctionalInterface
    private interface NativeDetectRequest {
        double[] run();
    }

    private List<NeuralNetworkPipeResult> flatToResults(double[] flat) {
        if (flat == null || flat.length == 0) {
            return List.of();
        }
        if (flat.length % 6 != 0) {
            logger.error("TensorRT JNI detector returned malformed detection array");
            return List.of();
        }

        List<NeuralNetworkPipeResult> results = new ArrayList<>(flat.length / 6);
        for (int i = 0; i < flat.length; i += 6) {
            double centerX = flat[i];
            double centerY = flat[i + 1];
            double rectWidth = flat[i + 2];
            double rectHeight = flat[i + 3];
            int classIdx = (int) Math.round(flat[i + 4]);
            double confidence = flat[i + 5];

            results.add(
                    new NeuralNetworkPipeResult(
                            new RotatedRect(new Point(centerX, centerY), new Size(rectWidth, rectHeight), 0.0),
                            classIdx,
                            confidence));
        }
        return results;
    }

    private boolean isValid() {
        return nativePtr != 0;
    }
}
