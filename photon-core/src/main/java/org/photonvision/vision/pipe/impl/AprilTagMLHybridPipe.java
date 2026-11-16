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

package org.photonvision.vision.pipe.impl;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;
import org.wpilib.vision.apriltag.AprilTagDetection;

/**
 * Composite pipe for ML-assisted AprilTag detection: runs YOLO ROI detection on the input frame,
 * pads each detected ROI, then runs the traditional WPILib AprilTag decoder on the matching regions
 * of the processed (single-channel) frame.
 *
 * <p>Each stage receives a different frame because each has different input requirements: the YOLO
 * model expects the full-fidelity source image matching its training distribution, while the WPILib
 * decoder requires a single-channel mat and performs its own adaptive thresholding internally.
 */
public class AprilTagMLHybridPipe
        extends CVPipe<Frame, MLDetectionResult, AprilTagMLHybridPipe.Params> implements Releasable {
    /**
     * Number of frames that can be in model inference at the same time. Each lane has its own copy of
     * the model. The default is 1: one frame at a time, on the vision thread.
     *
     * <p>More lanes help only when one inference takes longer than one camera frame period (for
     * example on an NPU that has more than one core). The pipeline then starts the inference of the
     * next frame while the inference of the last frame continues. Each frame gets the same
     * processing, so the tag data does not change.
     */
    private static final int INFERENCE_LANES =
            Math.max(1, Math.min(4, EnvUtils.integer("PV_MLTAG_INFERENCE_LANES", 1)));

    private static final AtomicInteger laneThreadNumber = new AtomicInteger();

    private final List<AprilTagROIDetectionPipe> lanePipes = new ArrayList<>();
    private final BlockingQueue<AprilTagROIDetectionPipe> freeLanes;
    private final ExecutorService laneExecutor;
    private final AprilTagROIDecodePipe roiDecodePipe = new AprilTagROIDecodePipe();
    private final double roiMergeIouThreshold = EnvUtils.decimal("PV_MLTAG_MERGE_ROI_IOU", 0.0);
    private final double maxRawRoiAreaFraction =
            EnvUtils.decimal("PV_MLTAG_MAX_RAW_ROI_AREA_FRAC", 0.0);

    // Moving average of the inference time of one frame. complete() writes it. One thread calls
    // complete() (the work thread of the pipeline), and the vision thread reads the values.
    private volatile double inferenceNanosAverage = 0;
    private volatile int inferenceSamples = 0;

    public static class Params {
        public final AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams;
        public final AprilTagROIDecodePipe.ROIDecodeParams decodeParams;
        public final int roiPaddingPixels;

        public Params(
                AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams,
                AprilTagROIDecodePipe.ROIDecodeParams decodeParams,
                int roiPaddingPixels) {
            this.detectionParams = detectionParams;
            this.decodeParams = decodeParams;
            this.roiPaddingPixels = roiPaddingPixels;
        }
    }

    /** A frame that is in model inference on a lane. */
    public static final class PendingDetection {
        public final Frame frame;
        private final Future<LaneOutput> future;

        private PendingDetection(Frame frame, Future<LaneOutput> future) {
            this.frame = frame;
            this.future = future;
        }

        /** Waits until the lane does not use the frame. After this the frame can be released. */
        public void await() {
            try {
                future.get();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } catch (ExecutionException e) {
                // The frame is not used after a failure.
            }
        }
    }

    /** The output of {@link #complete(PendingDetection)}. */
    public record CompletedDetection(
            MLDetectionResult result, long inferenceNanos, long decodeNanos) {
        public long nanosElapsed() {
            return inferenceNanos + decodeNanos;
        }
    }

    private record LaneOutput(List<RotatedRect> rois, long nanosElapsed) {}

    public AprilTagMLHybridPipe() {
        this(INFERENCE_LANES);
    }

    /**
     * @param lanes Number of frames that can be in model inference at the same time (1 to 4).
     */
    public AprilTagMLHybridPipe(int lanes) {
        int laneCount = Math.max(1, Math.min(4, lanes));
        for (int i = 0; i < laneCount; i++) {
            lanePipes.add(new AprilTagROIDetectionPipe());
        }
        if (laneCount > 1) {
            freeLanes = new ArrayBlockingQueue<>(laneCount, false, lanePipes);
            laneExecutor =
                    Executors.newFixedThreadPool(
                            laneCount,
                            runnable -> {
                                Thread thread =
                                        new Thread(runnable, "MLTag-Inference-" + laneThreadNumber.incrementAndGet());
                                thread.setDaemon(true);
                                return thread;
                            });
        } else {
            freeLanes = null;
            laneExecutor = null;
        }
    }

    /** Returns the number of frames that can be in model inference at the same time. */
    public int getLaneCount() {
        return lanePipes.size();
    }

    /**
     * Sets the parameters. The caller must make sure that no frame is in inference (each {@link
     * PendingDetection} is completed or awaited), because this can load a different model.
     */
    @Override
    public void setParams(Params newParams) {
        for (AprilTagROIDetectionPipe lane : lanePipes) {
            lane.setParams(newParams.detectionParams);
        }
        roiDecodePipe.setParams(newParams.decodeParams);
        super.setParams(newParams);
    }

    @Override
    protected MLDetectionResult process(Frame frame) {
        List<RotatedRect> rawRois = lanePipes.get(0).run(frame.colorImage).output;
        return decodeStage(frame, rawRois);
    }

    /**
     * Starts the model inference of a frame on a free lane and returns immediately. The frame must
     * stay valid until {@link #complete(PendingDetection)} or {@link PendingDetection#await()}
     * returns. Only available when the pipe has more than one lane.
     */
    public PendingDetection submit(Frame frame) {
        if (laneExecutor == null) {
            throw new IllegalStateException("This pipe has one inference lane");
        }
        final CVMat image = frame.colorImage;
        Future<LaneOutput> future =
                laneExecutor.submit(
                        () -> {
                            AprilTagROIDetectionPipe lane = freeLanes.take();
                            try {
                                long start = System.nanoTime();
                                List<RotatedRect> rois = lane.run(image).output;
                                return new LaneOutput(rois, System.nanoTime() - start);
                            } finally {
                                freeLanes.add(lane);
                            }
                        });
        return new PendingDetection(frame, future);
    }

    /**
     * Waits for the model inference of a frame, then decodes the tags in the regions that the model
     * found. This gives the same result as {@link #run(Object)} for that frame.
     */
    public CompletedDetection complete(PendingDetection pending)
            throws InterruptedException, ExecutionException {
        LaneOutput lane = pending.future.get();
        int samples = inferenceSamples;
        inferenceNanosAverage =
                samples == 0
                        ? lane.nanosElapsed()
                        : inferenceNanosAverage * 0.9 + lane.nanosElapsed() * 0.1;
        inferenceSamples = samples + 1;

        long decodeStart = System.nanoTime();
        MLDetectionResult result = decodeStage(pending.frame, lane.rois());
        return new CompletedDetection(result, lane.nanosElapsed(), System.nanoTime() - decodeStart);
    }

    /** Returns the moving average of the time of one model inference, in nanoseconds. */
    public double getInferenceNanosAverage() {
        return inferenceNanosAverage;
    }

    /** Returns the number of inferences that the average is made from. */
    public int getInferenceSamples() {
        return inferenceSamples;
    }

    private MLDetectionResult decodeStage(Frame frame, List<RotatedRect> rawRois) {
        if (rawRois == null || rawRois.isEmpty()) {
            return new MLDetectionResult(List.of(), List.of());
        }

        int frameWidth = imageWidth(frame);
        int frameHeight = imageHeight(frame);

        rawRois = filterRawRois(rawRois, frameWidth, frameHeight);
        if (rawRois.isEmpty()) {
            return new MLDetectionResult(List.of(), List.of());
        }

        List<RotatedRect> expandedRois = new ArrayList<>(rawRois.size());
        for (RotatedRect roi : rawRois) {
            expandedRois.add(
                    AprilTagROIDecodePipe.expandBbox(roi, params.roiPaddingPixels, frameWidth, frameHeight));
        }
        List<RotatedRect> decodeRois =
                mergeOverlappingRois(expandedRois, frameWidth, frameHeight, roiMergeIouThreshold);

        List<AprilTagDetection> decodedDetections = decodeRois(frame, decodeRois);

        return new MLDetectionResult(decodedDetections, decodeRois);
    }

    private static int imageWidth(Frame frame) {
        if (frame.colorImage.getNativeFrameHandle() != 0 && frame.frameStaticProperties != null) {
            return frame.frameStaticProperties.imageWidth;
        }
        return frame.colorImage.getMat().cols();
    }

    private static int imageHeight(Frame frame) {
        if (frame.colorImage.getNativeFrameHandle() != 0 && frame.frameStaticProperties != null) {
            return frame.frameStaticProperties.imageHeight;
        }
        return frame.colorImage.getMat().rows();
    }

    private List<AprilTagDetection> decodeRois(Frame frame, List<RotatedRect> decodeRois) {
        return decodeRoisOnJavaFrame(frame, decodeRois);
    }

    private List<AprilTagDetection> decodeRoisOnJavaFrame(Frame frame, List<RotatedRect> decodeRois) {
        var decodeSource =
                frame.processedImage.getMat().empty() ? frame.colorImage : frame.processedImage;
        AprilTagROIDecodePipe.ROIDecodeInput decodeInput =
                new AprilTagROIDecodePipe.ROIDecodeInput(decodeSource, decodeRois);
        CVPipe.CVPipeResult<List<AprilTagDetection>> decodeResult = roiDecodePipe.run(decodeInput);
        return decodeResult.output;
    }

    /**
     * @return true when the ROI detection stage has a loaded model (not {@code NullModel}).
     */
    public boolean isAvailable() {
        return lanePipes.get(0).isAvailable();
    }

    @Override
    public void release() {
        if (laneExecutor != null) {
            laneExecutor.shutdownNow();
        }
        for (AprilTagROIDetectionPipe lane : lanePipes) {
            lane.release();
        }
        roiDecodePipe.release();
    }

    private List<RotatedRect> filterRawRois(
            List<RotatedRect> rawRois, int frameWidth, int frameHeight) {
        if (maxRawRoiAreaFraction <= 0.0 || rawRois.isEmpty()) {
            return rawRois;
        }

        long framePixels = (long) frameWidth * frameHeight;
        double maxPixels = framePixels * maxRawRoiAreaFraction;
        List<RotatedRect> keptRois = new ArrayList<>(rawRois.size());

        for (int i = 0; i < rawRois.size(); i++) {
            RotatedRect roi = rawRois.get(i);
            Rect rect = clampRect(roi.boundingRect(), frameWidth, frameHeight);
            long roiPixels = rect.width <= 0 || rect.height <= 0 ? 0 : (long) rect.width * rect.height;
            if (roiPixels > maxPixels) {
                continue;
            }

            keptRois.add(roi);
        }

        return keptRois;
    }

    private static List<RotatedRect> mergeOverlappingRois(
            List<RotatedRect> rois, int frameWidth, int frameHeight, double iouThreshold) {
        if (iouThreshold <= 0.0 || rois.size() <= 1) {
            return rois;
        }

        List<Rect> mergedRects = new ArrayList<>();
        for (RotatedRect roi : rois) {
            Rect candidate = clampRect(roi.boundingRect(), frameWidth, frameHeight);
            if (candidate.width <= 0 || candidate.height <= 0) {
                continue;
            }

            boolean merged;
            do {
                merged = false;
                for (int i = 0; i < mergedRects.size(); i++) {
                    Rect existing = mergedRects.get(i);
                    if (rectIou(existing, candidate) >= iouThreshold) {
                        candidate = unionRect(existing, candidate);
                        mergedRects.remove(i);
                        merged = true;
                        break;
                    }
                }
            } while (merged);
            mergedRects.add(candidate);
        }

        if (mergedRects.size() == rois.size()) {
            return rois;
        }

        List<RotatedRect> mergedRois = new ArrayList<>(mergedRects.size());
        for (Rect rect : mergedRects) {
            mergedRois.add(
                    new RotatedRect(
                            new Point(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0),
                            new Size(rect.width, rect.height),
                            0.0));
        }
        return mergedRois;
    }

    private static Rect clampRect(Rect rect, int frameWidth, int frameHeight) {
        int x = Math.max(0, rect.x);
        int y = Math.max(0, rect.y);
        int w = Math.min(rect.width, frameWidth - x);
        int h = Math.min(rect.height, frameHeight - y);
        return new Rect(x, y, w, h);
    }

    private static double rectIou(Rect a, Rect b) {
        int left = Math.max(a.x, b.x);
        int top = Math.max(a.y, b.y);
        int right = Math.min(a.x + a.width, b.x + b.width);
        int bottom = Math.min(a.y + a.height, b.y + b.height);
        int intersectionWidth = Math.max(0, right - left);
        int intersectionHeight = Math.max(0, bottom - top);
        long intersection = (long) intersectionWidth * intersectionHeight;
        if (intersection == 0) {
            return 0.0;
        }
        long union = (long) a.width * a.height + (long) b.width * b.height - intersection;
        return union <= 0 ? 0.0 : (double) intersection / union;
    }

    private static Rect unionRect(Rect a, Rect b) {
        int x = Math.min(a.x, b.x);
        int y = Math.min(a.y, b.y);
        int right = Math.max(a.x + a.width, b.x + b.width);
        int bottom = Math.max(a.y + a.height, b.y + b.height);
        return new Rect(x, y, right - x, bottom - y);
    }
}
