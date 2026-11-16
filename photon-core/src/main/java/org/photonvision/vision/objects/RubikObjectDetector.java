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

import java.awt.Color;
import java.lang.ref.Cleaner;
import java.lang.ref.Cleaner.Cleanable;
import java.util.List;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.ColorHelper;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.rubik.RubikJNI;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

/** Manages an object detector using the rubik backend. */
public class RubikObjectDetector implements ObjectDetector {
    private static final Logger logger = new Logger(RubikObjectDetector.class, LogGroup.General);
    private static final boolean ENABLE_GSTREAMER_FRAME_STUB =
            EnvUtils.flag("PV_RUBIK_ENABLE_GSTREAMER_FRAME_STUB");

    private static final Cleaner cleaner = Cleaner.create();

    private final Cleanable cleanable;

    private static Runnable cleanupAction(long ptr) {
        return () -> RubikJNI.destroy(ptr);
    }

    /** Pointer to the native object */
    private final long ptr;

    private final RubikModel model;

    private final Size inputSize;

    // Scratch images for the model input. One vision thread uses one detector.
    private final Mat resizedScratch = new Mat();
    private final Mat letterboxed = new Mat();
    private final Mat letterboxedBgr = new Mat();

    /** Returns the model in use by this detector. */
    @Override
    public RubikModel getModel() {
        return model;
    }

    /**
     * Creates a new rubikObjectDetector from the given model.
     *
     * @param model The model to create the detector from.
     * @param inputSize The required image dimensions for the model. Images will be {@link
     *     Letterbox}ed to this shape.
     */
    public RubikObjectDetector(RubikModel model, Size inputSize) {
        this.model = model;
        this.inputSize = inputSize;

        // Create the detector
        try {
            ptr =
                    RubikJNI.create(
                            model.modelFile.getPath().toString(), model.properties.version().ordinal());
        } catch (Exception e) {
            logger.error("Failed to create detector from path " + model.modelFile.getPath(), e);
            throw new RuntimeException(
                    "Failed to create detector from path " + model.modelFile.getPath(), e);
        }

        if (!isValid()) {
            logger.error(
                    "Failed to create detector from path "
                            + model.modelFile.getPath()
                            + ". Please ensure the model is valid and compatible with the Rubik backend.");
            throw new RuntimeException(
                    "Failed to create detector from path " + model.modelFile.getPath());
        } else if (!RubikJNI.isQuantized(ptr)) {
            throw new UnsupportedOperationException("Model must be quantized.");
        }

        logger.debug("Created detector for model " + model.modelFile.getName());

        // Register the cleaner to release the detector when it goes out of scope
        cleanable = cleaner.register(this, cleanupAction(ptr));
    }

    /**
     * Returns the classes that the detector can detect
     *
     * @return The classes
     */
    @Override
    public List<String> getClasses() {
        return model.properties.labels();
    }

    /**
     * Detects objects in the given input image using the rubikDetector.
     *
     * @param in The input image to perform object detection on.
     * @param nmsThresh The threshold value for non-maximum suppression.
     * @param boxThresh The threshold value for bounding box detection.
     * @return A list of NeuralNetworkPipeResult objects representing the detected objects. Returns an
     *     empty list if the detector is not initialized or if no objects are detected.
     */
    @Override
    public List<NeuralNetworkPipeResult> detect(Mat in, double nmsThresh, double boxThresh) {
        if (!isValid()) {
            logger.error(
                    "Detector is not initialized, and so it can't be released! Model: "
                            + model.modelFile.getName());
            return null;
        }

        // Resize the frame to the input size of the model. The Mats are kept between calls, so
        // this step does not allocate memory for each frame.
        Letterbox scale =
                Letterbox.letterbox(
                        in, resizedScratch, letterboxed, this.inputSize, ColorHelper.colorToScalar(Color.GRAY));
        if (!letterboxed.size().equals(this.inputSize)) {
            throw new RuntimeException("Letterboxed frame is not the right size!");
        }

        // A monochrome camera gives a 1-channel image. The model input has 3 channels. Copy the
        // gray value into each channel after the resize. This gives the same bytes as a resize of
        // a 3-channel image that has equal channels, and the resize has one third of the work.
        Mat modelInput = letterboxed;
        if (letterboxed.channels() == 1) {
            Imgproc.cvtColor(letterboxed, letterboxedBgr, Imgproc.COLOR_GRAY2BGR);
            modelInput = letterboxedBgr;
        }

        // Detect objects in the letterboxed frame
        var results = RubikJNI.detect(ptr, modelInput.getNativeObjAddr(), boxThresh, nmsThresh);

        if (results == null) {
            return List.of();
        }

        return scale.resizeDetections(
                List.of(results).stream()
                        .map(it -> new NeuralNetworkPipeResult(it.rect, it.class_id, it.conf))
                        .toList());
    }

    @Override
    public List<NeuralNetworkPipeResult> detect(CVMat in, double nmsThresh, double boxThresh) {
        if (in.getNativeFrameHandle() != 0 && ENABLE_GSTREAMER_FRAME_STUB) {
            logger.warn(
                    "Rubik received a native GStreamer frame, but rubik_jni does not yet expose a"
                            + " GstBuffer/DMABUF/QNN entrypoint. Falling back to Mat letterbox path.");
        }

        return detect(in.getMat(), nmsThresh, boxThresh);
    }

    /** Thread-safe method to release the detector. */
    @Override
    public void release() {
        cleanable.clean();
        resizedScratch.release();
        letterboxed.release();
        letterboxedBgr.release();
    }

    private boolean isValid() {
        return ptr != 0;
    }
}
