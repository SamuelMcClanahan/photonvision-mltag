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
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.ColorHelper;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.rknn.RknnJNI;
import org.photonvision.vision.pipe.impl.NeuralNetworkPipeResult;

/** Manages an object detector using the rknn backend. */
public class RknnObjectDetector implements ObjectDetector {
    private static final Logger logger = new Logger(RknnObjectDetector.class, LogGroup.General);

    private static final Cleaner cleaner = Cleaner.create();

    private final Cleanable cleanable;

    private static Runnable cleanupAction(long ptr) {
        return () -> RknnJNI.destroy(ptr);
    }

    /**
     * Returns the NPU core selection for new detectors. PV_RKNN_CORE_MASK can be: -1 (the runtime
     * selects a core that is not busy, default), 0, 1 or 2 (one specified core), 10 (cores 0 and 1
     * together), 210 (all three cores together). This only changes where the model runs. The model
     * output does not change.
     */
    private static int npuCoreSelection() {
        int requested = EnvUtils.integer("PV_RKNN_CORE_MASK", -1);
        return switch (requested) {
            case -1, 0, 1, 2, 10, 210 -> requested;
            default -> -1;
        };
    }

    /** Pointer to the native object */
    private final long objPointer;

    private final RknnModel model;

    private final Size inputSize;

    // Scratch images for the model input. One vision thread uses one detector.
    private final Letterbox.ModelInputScratch scratch = new Letterbox.ModelInputScratch();

    /** Returns the model in use by this detector. */
    @Override
    public RknnModel getModel() {
        return model;
    }

    /**
     * Creates a new RknnObjectDetector from the given model.
     *
     * @param model The model to create the detector from.
     * @param inputSize The required image dimensions for the model. Images will be {@link
     *     Letterbox}ed to this shape.
     */
    public RknnObjectDetector(RknnModel model, Size inputSize) {
        this.model = model;
        this.inputSize = inputSize;

        // Create the detector
        objPointer =
                RknnJNI.create(
                        model.modelFile.getPath(),
                        model.properties.labels().size(),
                        model.properties.version().ordinal(),
                        npuCoreSelection());
        if (objPointer <= 0) {
            throw new RuntimeException(
                    "Failed to create detector from path " + model.modelFile.getPath());
        } else if (!RknnJNI.isQuantized(objPointer)) {
            throw new UnsupportedOperationException("Model must be quantized.");
        }

        logger.debug("Created detector for model " + model.modelFile.getName());

        // Register the cleaner to release the detector when it goes out of scope
        cleanable = cleaner.register(this, cleanupAction(objPointer));
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
     * Detects objects in the given input image using the RknnDetector.
     *
     * @param in The input image to perform object detection on.
     * @param nmsThresh The threshold value for non-maximum suppression.
     * @param boxThresh The threshold value for bounding box detection.
     * @return A list of NeuralNetworkPipeResult objects representing the detected objects. Returns an
     *     empty list if the detector is not initialized or if no objects are detected.
     */
    @Override
    public List<NeuralNetworkPipeResult> detect(Mat in, double nmsThresh, double boxThresh) {
        if (objPointer <= 0) {
            // Report error and make sure to include the model name
            logger.error("Detector is not initialized! Model: " + model.modelFile.getName());
            return List.of();
        }

        // Resize the frame to the input size of the model. The Mats are kept between calls, so
        // this step does not allocate memory for each frame. A monochrome camera gives a 1-channel
        // image. The fast path for it is used only when the self-check proved on this OpenCV build
        // that it gives the same bytes as the base operation. Otherwise the base operation runs.
        Letterbox.ModelInput input =
                Letterbox.prepareModelInput(
                        in, scratch, this.inputSize, ColorHelper.colorToScalar(Color.GRAY));
        if (!input.image().size().equals(this.inputSize)) {
            throw new RuntimeException("Letterboxed frame is not the right size!");
        }

        // Detect objects in the letterboxed frame
        var results =
                RknnJNI.detect(objPointer, input.image().getNativeObjAddr(), nmsThresh, boxThresh);

        if (results == null) {
            return List.of();
        }

        return input
                .scale()
                .resizeDetections(
                        List.of(results).stream()
                                .map(it -> new NeuralNetworkPipeResult(it.rect, it.class_id, it.conf))
                                .toList());
    }

    /** Thread-safe method to release the detector. */
    @Override
    public void release() {
        cleanable.clean();
        scratch.release();
    }
}
