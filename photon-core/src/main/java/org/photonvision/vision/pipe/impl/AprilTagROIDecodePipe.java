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
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.RotatedRect;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.Releasable;
import org.photonvision.vision.pipe.CVPipe;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagDetector;

/**
 * Pipe that decodes AprilTags within ROIs detected by ML. For each ROI from ML detection, it
 * extracts a sub-image, runs the traditional WPILib AprilTag detector, and maps the detected corner
 * coordinates and homography back to full-frame coordinates.
 *
 * <p>Both corner coordinates AND the homography matrix must be transformed from ROI coordinates to
 * full-frame coordinates.
 */
public class AprilTagROIDecodePipe
        extends CVPipe<
                AprilTagROIDecodePipe.ROIDecodeInput,
                List<AprilTagDetection>,
                AprilTagROIDecodePipe.ROIDecodeParams>
        implements Releasable {
    private static final Logger logger =
            new Logger(AprilTagROIDecodePipe.class, LogGroup.VisionModule);

    /**
     * Decode different ROIs on different threads. This is on by default. Set
     * PV_MLTAG_PARALLEL_ROI_DECODE=false to decode all ROIs on the vision thread.
     */
    private static final boolean PARALLEL_ROI_DECODE =
            EnvUtils.flag("PV_MLTAG_PARALLEL_ROI_DECODE", true);

    /**
     * Maximum number of ROIs that one pipe decodes at the same time. 0 = use the "threads" setting of
     * the AprilTag pipeline.
     */
    private static final int PARALLEL_ROI_WORKERS =
            Math.max(0, EnvUtils.integer("PV_MLTAG_PARALLEL_ROI_WORKERS", 0));

    /** Threads inside each detector of a parallel worker. One ROI is small, so the default is 1. */
    private static final int PARALLEL_DETECTOR_THREADS =
            Math.max(1, EnvUtils.integer("PV_MLTAG_PARALLEL_ROI_DETECTOR_THREADS", 1));

    private static final int PARALLEL_ROI_MIN_COUNT =
            Math.max(2, EnvUtils.integer("PV_MLTAG_PARALLEL_ROI_MIN_COUNT", 2));

    /**
     * A single ROI with this number of pixels or more is decoded with the multi-thread detector. A
     * smaller ROI is decoded with a single-thread detector, because the cost to start the detector
     * threads is then larger than the gain.
     */
    private static final int LARGE_ROI_PIXELS =
            Math.max(0, EnvUtils.integer("PV_MLTAG_LARGE_ROI_PIXELS", 160_000));

    private static final boolean CONTIGUOUS_ROI_COPY = EnvUtils.flag("PV_MLTAG_CONTIGUOUS_ROI_COPY");

    /** Hard limit for the number of detectors that one pipe makes. Each one uses about 19 MB. */
    private static final int MAX_WORKERS_PER_PIPE = 8;

    /**
     * One pool for all cameras. With two cameras, two private pools would start more threads than the
     * processor has cores. Each pipe also decodes one group of ROIs on its own thread.
     */
    private static final class SharedPool {
        static final int THREADS =
                Math.max(
                        1,
                        EnvUtils.integer(
                                "PV_MLTAG_ROI_POOL_THREADS",
                                Math.min(MAX_WORKERS_PER_PIPE, Runtime.getRuntime().availableProcessors()) - 1));
        static final AtomicInteger threadNumber = new AtomicInteger();
        static final ExecutorService EXECUTOR =
                Executors.newFixedThreadPool(
                        THREADS,
                        runnable -> {
                            Thread thread =
                                    new Thread(runnable, "AprilTag-ROI-Decode-" + threadNumber.incrementAndGet());
                            thread.setDaemon(true);
                            return thread;
                        });
    }

    /** Context object containing ATR scaling information for a single ROI. */
    private static class ATRContext {
        final double scaleFactor;
        final boolean wasScaled;

        ATRContext(double scaleFactor) {
            this.scaleFactor = scaleFactor;
            this.wasScaled = scaleFactor < 1.0;
        }

        static ATRContext noScaling() {
            return new ATRContext(1.0);
        }
    }

    /** Input container for ROI decode pipe */
    public static class ROIDecodeInput {
        public final CVMat grayFrame;
        public final List<RotatedRect> rois;

        public ROIDecodeInput(CVMat grayFrame, List<RotatedRect> rois) {
            this.grayFrame = grayFrame;
            this.rois = rois;
        }
    }

    /** Parameters for ROI decode pipe */
    public static class ROIDecodeParams {
        public AprilTagFamily tagFamily = AprilTagFamily.kTag36h11;
        public AprilTagDetector.Config detectorConfig;
        public AprilTagDetector.QuadThresholdParameters quadParams;
        public int maxHammingDistance = 0;
        public double minDecisionMargin = 35;

        // Adaptive Tag Resizing (ATR) parameters
        /** Enable ATR (adaptive tag resizing) */
        public boolean atrEnabled = true;

        /** Target dimension for ATR - tags wider than this will be downscaled */
        public int atrTargetDimension = 160;

        /** Minimum scale factor for ATR (prevents extreme downscaling) */
        public double atrMinScaleFactor = 0.25;

        public ROIDecodeParams() {
            detectorConfig = new AprilTagDetector.Config();
            detectorConfig.numThreads = 1;
            detectorConfig.quadDecimate = 1;
            quadParams = new AprilTagDetector.QuadThresholdParameters();
            // Match the defaults from AprilTagPipeline
            quadParams.minClusterPixels = 5;
            quadParams.maxNumMaxima = 10;
            quadParams.criticalAngle = 45 * Math.PI / 180.0;
            quadParams.maxLineFitMSE = 10.0f;
            quadParams.minWhiteBlackDiff = 5;
            quadParams.deglitch = false;
        }
    }

    /** The detector with the thread count from the pipeline settings. Used for one large ROI. */
    private AprilTagDetector detector;

    private AprilTagFamily currentFamily;
    private final Mat resizedRoi = new Mat();
    private final Mat grayRoi = new Mat();
    private final Mat contiguousRoi = new Mat();

    /**
     * Single-thread detectors, one for each group of ROIs that can run at the same time. They are
     * made when they are first necessary, because each detector uses much memory.
     */
    private final List<ParallelWorker> parallelWorkers = new ArrayList<>();

    private int maxParallelWorkers = 1;

    public AprilTagROIDecodePipe() {
        detector = new AprilTagDetector();
    }

    @Override
    protected List<AprilTagDetection> process(ROIDecodeInput input) {
        if (input.grayFrame == null
                || input.grayFrame.getMat().empty()
                || input.rois == null
                || input.rois.isEmpty()) {
            return List.of();
        }

        Mat fullFrame = input.grayFrame.getMat();
        int frameWidth = fullFrame.cols();
        int frameHeight = fullFrame.rows();

        if (isParallelDecodeActive(input.rois.size())) {
            return processParallel(input.rois, fullFrame, frameWidth, frameHeight);
        }

        return processSerial(input.rois, fullFrame, frameWidth, frameHeight);
    }

    /**
     * Decodes all ROIs on the calling thread. A large ROI uses the detector that has the thread count
     * from the pipeline settings. A small ROI uses a single-thread detector.
     */
    private List<AprilTagDetection> processSerial(
            List<RotatedRect> rois, Mat fullFrame, int frameWidth, int frameHeight) {
        List<AprilTagDetection> allDetections = new ArrayList<>();
        for (RotatedRect roi : rois) {
            AprilTagDetector activeDetector = detector;
            if (useSingleThreadDetector(roi, frameWidth, frameHeight)) {
                activeDetector = getWorker(0).detector;
            }
            RoiDecodeResult roiResult =
                    decodeOneRoi(
                            roi,
                            fullFrame,
                            frameWidth,
                            frameHeight,
                            activeDetector,
                            grayRoi,
                            resizedRoi,
                            contiguousRoi);
            allDetections.addAll(roiResult.detections);
        }

        return deduplicateByTagId(allDetections);
    }

    private boolean useSingleThreadDetector(RotatedRect roi, int frameWidth, int frameHeight) {
        if (!PARALLEL_ROI_DECODE || params == null || params.detectorConfig.numThreads <= 1) {
            return false;
        }
        return roiPixelArea(roi, frameWidth, frameHeight) < LARGE_ROI_PIXELS;
    }

    private boolean isParallelDecodeActive(int roiCount) {
        return PARALLEL_ROI_DECODE && maxParallelWorkers > 1 && roiCount >= PARALLEL_ROI_MIN_COUNT;
    }

    /**
     * Decodes groups of ROIs at the same time. The ROIs are put into groups that have almost the same
     * number of pixels. The calling thread decodes one group, and the shared pool decodes the other
     * groups. The result has the same detections in the same order as {@link #processSerial}.
     */
    private List<AprilTagDetection> processParallel(
            List<RotatedRect> rois, Mat fullFrame, int frameWidth, int frameHeight) {
        int groupCount = Math.min(maxParallelWorkers, rois.size());
        List<List<Integer>> groups = balanceRoisByArea(rois, frameWidth, frameHeight, groupCount);

        @SuppressWarnings("unchecked")
        List<AprilTagDetection>[] detectionsByRoi = new List[rois.size()];

        List<Future<?>> futures = new ArrayList<>(groupCount - 1);
        try {
            for (int groupIndex = 1; groupIndex < groupCount; groupIndex++) {
                final List<Integer> group = groups.get(groupIndex);
                final ParallelWorker worker = getWorker(groupIndex);
                futures.add(
                        SharedPool.EXECUTOR.submit(
                                () ->
                                        decodeRoiGroup(
                                                rois, group, fullFrame, frameWidth, frameHeight, worker, detectionsByRoi)));
            }

            decodeRoiGroup(
                    rois, groups.get(0), fullFrame, frameWidth, frameHeight, getWorker(0), detectionsByRoi);

            for (Future<?> future : futures) {
                future.get();
            }
        } catch (Exception e) {
            // Wait for the tasks that still run. They use the frame, and the caller releases the
            // frame after this method returns.
            for (Future<?> future : futures) {
                try {
                    future.get();
                } catch (Exception ignored) {
                    // The serial decode below gives the result.
                }
            }
            if (e instanceof InterruptedException) {
                Thread.currentThread().interrupt();
            }
            logger.warn("Parallel ROI decode failed, falling back to serial decode: " + e.getMessage());
            return processSerial(rois, fullFrame, frameWidth, frameHeight);
        }

        // Join the results in the ROI order, so the order does not depend on thread timing.
        List<AprilTagDetection> allDetections = new ArrayList<>();
        for (List<AprilTagDetection> roiDetections : detectionsByRoi) {
            if (roiDetections != null) {
                allDetections.addAll(roiDetections);
            }
        }
        return deduplicateByTagId(allDetections);
    }

    /**
     * Puts the ROI indexes into groups. The largest ROI goes first, and each ROI goes to the group
     * that has the smallest pixel total at that time.
     */
    private List<List<Integer>> balanceRoisByArea(
            List<RotatedRect> rois, int frameWidth, int frameHeight, int groupCount) {
        List<List<Integer>> groups = new ArrayList<>(groupCount);
        long[] groupPixels = new long[groupCount];
        for (int i = 0; i < groupCount; i++) {
            groups.add(new ArrayList<>());
        }

        long[] areas = new long[rois.size()];
        List<Integer> order = new ArrayList<>(rois.size());
        for (int i = 0; i < rois.size(); i++) {
            areas[i] = roiPixelArea(rois.get(i), frameWidth, frameHeight);
            order.add(i);
        }
        order.sort((a, b) -> Long.compare(areas[b], areas[a]));

        for (int roiIndex : order) {
            int lightestGroup = 0;
            for (int i = 1; i < groupCount; i++) {
                if (groupPixels[i] < groupPixels[lightestGroup]) {
                    lightestGroup = i;
                }
            }
            groups.get(lightestGroup).add(roiIndex);
            groupPixels[lightestGroup] += areas[roiIndex];
        }
        return groups;
    }

    private void decodeRoiGroup(
            List<RotatedRect> rois,
            List<Integer> group,
            Mat fullFrame,
            int frameWidth,
            int frameHeight,
            ParallelWorker worker,
            List<AprilTagDetection>[] detectionsByRoi) {
        for (int roiIndex : group) {
            RoiDecodeResult roiResult =
                    decodeOneRoi(
                            rois.get(roiIndex),
                            fullFrame,
                            frameWidth,
                            frameHeight,
                            worker.detector,
                            worker.grayRoi,
                            worker.resizedRoi,
                            worker.contiguousRoi);
            detectionsByRoi[roiIndex] = roiResult.detections;
        }
    }

    private RoiDecodeResult decodeOneRoi(
            RotatedRect roi,
            Mat fullFrame,
            int frameWidth,
            int frameHeight,
            AprilTagDetector activeDetector,
            Mat grayScratch,
            Mat resizeScratch,
            Mat contiguousScratch) {
        RoiDecodeResult result = new RoiDecodeResult();

        // ROIs are already expanded by the caller (AprilTagMLHybridPipe)
        Rect roiRect = toIntRect(roi);

        if (roiRect.width <= 0 || roiRect.height <= 0) {
            return result;
        }

        // Clamp to frame bounds (defensive)
        roiRect = clampToFrame(roiRect, frameWidth, frameHeight);
        if (roiRect.width <= 0 || roiRect.height <= 0) {
            return result;
        }

        // Extract full-resolution submat
        Mat roiMat = fullFrame.submat(roiRect);
        try {
            return decodeOneCroppedRoi(
                    roiRect,
                    roiMat,
                    frameWidth,
                    frameHeight,
                    activeDetector,
                    grayScratch,
                    resizeScratch,
                    contiguousScratch,
                    result);
        } finally {
            roiMat.release();
        }
    }

    private RoiDecodeResult decodeOneCroppedRoi(
            Rect roiRect,
            Mat roiMat,
            int frameWidth,
            int frameHeight,
            AprilTagDetector activeDetector,
            Mat grayScratch,
            Mat resizeScratch,
            Mat contiguousScratch) {
        RoiDecodeResult result = new RoiDecodeResult();
        roiRect = clampToFrame(roiRect, frameWidth, frameHeight);
        if (roiRect.width <= 0 || roiRect.height <= 0 || roiMat.empty()) {
            return result;
        }
        return decodeOneCroppedRoi(
                roiRect,
                roiMat,
                frameWidth,
                frameHeight,
                activeDetector,
                grayScratch,
                resizeScratch,
                contiguousScratch,
                result);
    }

    private RoiDecodeResult decodeOneCroppedRoi(
            Rect roiRect,
            Mat roiMat,
            int frameWidth,
            int frameHeight,
            AprilTagDetector activeDetector,
            Mat grayScratch,
            Mat resizeScratch,
            Mat contiguousScratch,
            RoiDecodeResult result) {
        Mat decodeRoiMat = roiMat;
        if (roiMat.channels() > 1) {
            Imgproc.cvtColor(roiMat, grayScratch, Imgproc.COLOR_BGR2GRAY);
            decodeRoiMat = grayScratch;
        }

        // === STAGE 1: COARSE - Calculate scaling and resize ===
        ATRContext atrContext;
        Mat processingMat;

        if (params.atrEnabled && roiRect.width > params.atrTargetDimension) {
            // Calculate scale factor: S = min(1.0, T_dim / w)
            double S = (double) params.atrTargetDimension / roiRect.width;
            S = Math.max(S, params.atrMinScaleFactor); // Clamp to prevent extreme scaling

            int scaledWidth = (int) Math.round(roiRect.width * S);
            int scaledHeight = (int) Math.round(roiRect.height * S);

            atrContext = new ATRContext(S);

            // Downscale ROI to target dimension
            Imgproc.resize(
                    decodeRoiMat,
                    resizeScratch,
                    new Size(scaledWidth, scaledHeight),
                    0,
                    0,
                    Imgproc.INTER_AREA); // INTER_AREA best for downscaling
            processingMat = resizeScratch;
        } else {
            // No scaling needed (tag already small or ATR disabled)
            atrContext = ATRContext.noScaling();
            processingMat = decodeRoiMat;
        }

        if (CONTIGUOUS_ROI_COPY && !processingMat.isContinuous()) {
            processingMat.copyTo(contiguousScratch);
            processingMat = contiguousScratch;
        }

        // Run AprilTag detection on (possibly scaled) image
        AprilTagDetection[] roiDetections = activeDetector.detect(processingMat);

        for (AprilTagDetection det : roiDetections) {
            if (det.getHamming() > params.maxHammingDistance) continue;
            if (det.getDecisionMargin() < params.minDecisionMargin) continue;

            // === STAGE 2: PROJECTION - Map coordinates back to full-frame ===
            AprilTagDetection mappedDetection;
            if (atrContext.wasScaled) {
                mappedDetection = mapToFullFrameWithATR(det, roiRect, atrContext);
            } else {
                mappedDetection = mapToFullFrame(det, roiRect);
            }

            result.detections.add(mappedDetection);
        }

        return result;
    }

    /**
     * Maps detection coordinates from ROI space to full-frame space.
     *
     * <p>Both corners AND homography must be transformed. The pose estimator uses the homography
     * matrix internally for pose estimation, not just the corners.
     *
     * @param det The detection in ROI coordinates
     * @param roiOffset The ROI rectangle defining the offset from full frame origin
     * @return A new AprilTagDetection with coordinates mapped to full frame
     */
    private AprilTagDetection mapToFullFrame(AprilTagDetection det, Rect roiOffset) {
        // Map all 4 corners from ROI coordinates to full-frame coordinates
        double[] mappedCorners = new double[8];
        for (int i = 0; i < 4; i++) {
            mappedCorners[i * 2] = det.getCornerX(i) + roiOffset.x;
            mappedCorners[i * 2 + 1] = det.getCornerY(i) + roiOffset.y;
        }

        // Map center coordinate
        double centerX = det.getCenterX() + roiOffset.x;
        double centerY = det.getCenterY() + roiOffset.y;

        // Transform homography from ROI coordinates to full-frame coordinates
        double[] transformedHomography =
                transformHomography(det.getHomography(), roiOffset.x, roiOffset.y);

        return new AprilTagDetection(
                det.getFamily(),
                det.getId(),
                det.getHamming(),
                det.getDecisionMargin(),
                transformedHomography,
                centerX,
                centerY,
                mappedCorners);
    }

    /**
     * Maps detection coordinates from scaled ROI space to full-frame space. Handles both translation
     * AND scaling transformations.
     *
     * @param det The detection in scaled ROI coordinates
     * @param roiOffset The ROI rectangle defining the offset from full frame origin
     * @param atrContext The ATR scaling context
     * @return A new AprilTagDetection with coordinates mapped to full frame
     */
    private AprilTagDetection mapToFullFrameWithATR(
            AprilTagDetection det, Rect roiOffset, ATRContext atrContext) {
        double invS = 1.0 / atrContext.scaleFactor;

        // Map all 4 corners from scaled ROI coordinates to full-frame coordinates
        // x_full = (x_scaled / S) + roi_x
        double[] mappedCorners = new double[8];
        for (int i = 0; i < 4; i++) {
            mappedCorners[i * 2] = det.getCornerX(i) * invS + roiOffset.x;
            mappedCorners[i * 2 + 1] = det.getCornerY(i) * invS + roiOffset.y;
        }

        // Map center coordinate
        double centerX = det.getCenterX() * invS + roiOffset.x;
        double centerY = det.getCenterY() * invS + roiOffset.y;

        // Transform homography with scaling
        double[] transformedHomography =
                transformHomographyWithScale(
                        det.getHomography(), roiOffset.x, roiOffset.y, atrContext.scaleFactor);

        return new AprilTagDetection(
                det.getFamily(),
                det.getId(),
                det.getHamming(),
                det.getDecisionMargin(),
                transformedHomography,
                centerX,
                centerY,
                mappedCorners);
    }

    /**
     * Transforms a homography matrix from ROI coordinates to full-frame coordinates.
     *
     * <p>The homography H satisfies: [x_roi, y_roi, 1]^T ~ H * [X_tag, Y_tag, 1]^T
     *
     * <p>To convert to full-frame coordinates where x_full = x_roi + offsetX and y_full = y_roi +
     * offsetY, we compute H_full = T * H_roi where T is the translation matrix:
     *
     * <pre>
     * | 1  0  offsetX |
     * | 0  1  offsetY |
     * | 0  0  1       |
     * </pre>
     *
     * <p>The UMich AprilTag library stores homography as row-major 3x3: [h00, h01, h02, h10, h11,
     * h12, h20, h21, h22] source: https://april.eecs.umich.edu/media/pdfs/olson2011tags.pdf
     *
     * @param h The original homography (9 elements, row-major 3x3)
     * @param offsetX The x offset from ROI origin to full-frame origin
     * @param offsetY The y offset from ROI origin to full-frame origin
     * @return The transformed homography in full-frame coordinates
     */
    private double[] transformHomography(double[] h, int offsetX, int offsetY) {
        // T * H where T = [[1,0,tx],[0,1,ty],[0,0,1]]
        // (T*H)[0][j] = 1*H[0][j] + 0*H[1][j] + tx*H[2][j] = H[0][j] + tx*H[2][j]
        // (T*H)[1][j] = 0*H[0][j] + 1*H[1][j] + ty*H[2][j] = H[1][j] + ty*H[2][j]
        // (T*H)[2][j] = 0*H[0][j] + 0*H[1][j] + 1*H[2][j] = H[2][j]

        double[] result = new double[9];
        // Row 0: H[0][j] + offsetX * H[2][j]
        result[0] = h[0] + offsetX * h[6];
        result[1] = h[1] + offsetX * h[7];
        result[2] = h[2] + offsetX * h[8];
        // Row 1: H[1][j] + offsetY * H[2][j]
        result[3] = h[3] + offsetY * h[6];
        result[4] = h[4] + offsetY * h[7];
        result[5] = h[5] + offsetY * h[8];
        // Row 2: unchanged
        result[6] = h[6];
        result[7] = h[7];
        result[8] = h[8];
        return result;
    }

    /**
     * Transforms a homography matrix from scaled ROI coordinates to full-frame coordinates.
     *
     * <p>The transformation combines inverse scaling and translation: H_full = T * S_inv * H_scaled
     *
     * <p>Where: - S_inv = [[1/S, 0, 0], [0, 1/S, 0], [0, 0, 1]] (inverse scale) - T = [[1, 0,
     * offsetX], [0, 1, offsetY], [0, 0, 1]] (translation)
     *
     * <p>Combined formula: Row 0: H[0..2] / S + offsetX * H[6..8] Row 1: H[3..5] / S + offsetY *
     * H[6..8] Row 2: unchanged (H[6..8])
     *
     * @param h The original homography (9 elements, row-major 3x3)
     * @param offsetX The x offset from ROI origin to full-frame origin
     * @param offsetY The y offset from ROI origin to full-frame origin
     * @param S The scale factor used for ROI resizing (< 1 means downscaled)
     * @return The transformed homography in full-frame coordinates
     */
    private double[] transformHomographyWithScale(double[] h, int offsetX, int offsetY, double S) {
        double invS = 1.0 / S;
        double[] result = new double[9];

        // Row 0: H[i]/S + offsetX * H[6+j]
        result[0] = h[0] * invS + offsetX * h[6];
        result[1] = h[1] * invS + offsetX * h[7];
        result[2] = h[2] * invS + offsetX * h[8];

        // Row 1: H[i]/S + offsetY * H[6+j]
        result[3] = h[3] * invS + offsetY * h[6];
        result[4] = h[4] * invS + offsetY * h[7];
        result[5] = h[5] * invS + offsetY * h[8];

        // Row 2: unchanged
        result[6] = h[6];
        result[7] = h[7];
        result[8] = h[8];

        return result;
    }

    /**
     * Expands a bounding box by adding fixed pixel padding on each side, then clamping to image
     * bounds. Additive padding naturally provides more relative expansion for small/far tags and less
     * for large/close tags.
     *
     * @param bbox Original bounding box
     * @param paddingPixels Number of pixels to add on each side of the bounding box
     * @param imageWidth Image width for clamping
     * @param imageHeight Image height for clamping
     * @return Expanded and clamped bounding box
     */
    public static RotatedRect expandBbox(
            RotatedRect bbox, int paddingPixels, int imageWidth, int imageHeight) {
        double newWidth = bbox.size.width + 2.0 * paddingPixels;
        double newHeight = bbox.size.height + 2.0 * paddingPixels;

        // Clamp center so the (axis-aligned) half-extents stay within frame
        double halfW = newWidth / 2.0;
        double halfH = newHeight / 2.0;
        double cx = Math.max(halfW, Math.min(bbox.center.x, imageWidth - halfW));
        double cy = Math.max(halfH, Math.min(bbox.center.y, imageHeight - halfH));

        return new RotatedRect(new Point(cx, cy), new Size(newWidth, newHeight), bbox.angle);
    }

    /** Converts a RotatedRect to integer Rect (axis-aligned bounding rectangle). */
    private Rect toIntRect(RotatedRect r) {
        return r.boundingRect();
    }

    private long roiPixelArea(RotatedRect roi, int frameWidth, int frameHeight) {
        Rect roiRect = clampToFrame(toIntRect(roi), frameWidth, frameHeight);
        if (roiRect.width <= 0 || roiRect.height <= 0) {
            return 0;
        }
        return (long) roiRect.width * roiRect.height;
    }

    /** Clamps a rectangle to frame bounds. */
    private Rect clampToFrame(Rect r, int frameWidth, int frameHeight) {
        int x = Math.max(0, r.x);
        int y = Math.max(0, r.y);
        int w = Math.min(r.width, frameWidth - x);
        int h = Math.min(r.height, frameHeight - y);
        return new Rect(x, y, w, h);
    }

    /**
     * Deduplicates detections by tag ID, keeping the one with highest decision margin. This handles
     * cases where overlapping ROIs detect the same tag.
     */
    private List<AprilTagDetection> deduplicateByTagId(List<AprilTagDetection> detections) {
        if (detections.size() <= 1) {
            return detections;
        }

        Map<Integer, AprilTagDetection> bestByTagId = new HashMap<>();

        for (AprilTagDetection det : detections) {
            int tagId = det.getId();
            AprilTagDetection existing = bestByTagId.get(tagId);

            if (existing == null || det.getDecisionMargin() > existing.getDecisionMargin()) {
                bestByTagId.put(tagId, det);
            }
        }

        return new ArrayList<>(bestByTagId.values());
    }

    @Override
    public void setParams(ROIDecodeParams newParams) {
        if (newParams.tagFamily != currentFamily) {
            detector.clearFamilies();
            detector.addFamily(newParams.tagFamily.getNativeName());
            // The single-thread detectors are made again with the new family when they are necessary.
            releaseParallelWorkers();
            currentFamily = newParams.tagFamily;
        }

        detector.setConfig(newParams.detectorConfig);
        detector.setQuadThresholdParameters(newParams.quadParams);

        int requestedWorkers =
                PARALLEL_ROI_WORKERS > 0
                        ? PARALLEL_ROI_WORKERS
                        : Math.max(1, newParams.detectorConfig.numThreads);
        // The calling thread decodes one group, so the pool can serve THREADS more groups.
        maxParallelWorkers =
                Math.max(
                        1, Math.min(requestedWorkers, Math.min(MAX_WORKERS_PER_PIPE, SharedPool.THREADS + 1)));

        super.setParams(newParams);

        for (ParallelWorker worker : parallelWorkers) {
            configureWorker(worker);
        }
    }

    @Override
    public void release() {
        resizedRoi.release();
        grayRoi.release();
        contiguousRoi.release();
        if (detector != null) {
            detector.close();
            detector = null;
        }
        releaseParallelWorkers();
    }

    private void releaseParallelWorkers() {
        for (ParallelWorker worker : parallelWorkers) {
            worker.release();
        }
        parallelWorkers.clear();
    }

    /** Returns the worker for a group index. Only the vision thread calls this method. */
    private ParallelWorker getWorker(int index) {
        while (parallelWorkers.size() <= index) {
            ParallelWorker worker = new ParallelWorker();
            if (currentFamily != null) {
                worker.detector.addFamily(currentFamily.getNativeName());
            }
            configureWorker(worker);
            parallelWorkers.add(worker);
        }
        return parallelWorkers.get(index);
    }

    /**
     * Gives a worker detector the same settings as the main detector. Only the thread count is
     * different. The thread count does not change which quads the detector finds.
     */
    private void configureWorker(ParallelWorker worker) {
        if (params == null) {
            return;
        }
        AprilTagDetector.Config workerConfig = copyConfig(params.detectorConfig);
        workerConfig.numThreads = PARALLEL_DETECTOR_THREADS;
        worker.detector.setConfig(workerConfig);
        worker.detector.setQuadThresholdParameters(copyQuadParams(params.quadParams));
    }

    private static AprilTagDetector.Config copyConfig(AprilTagDetector.Config config) {
        AprilTagDetector.Config copy = new AprilTagDetector.Config();
        copy.numThreads = config.numThreads;
        copy.quadDecimate = config.quadDecimate;
        copy.quadSigma = config.quadSigma;
        copy.refineEdges = config.refineEdges;
        copy.decodeSharpening = config.decodeSharpening;
        copy.debug = config.debug;
        return copy;
    }

    private static AprilTagDetector.QuadThresholdParameters copyQuadParams(
            AprilTagDetector.QuadThresholdParameters quadParams) {
        AprilTagDetector.QuadThresholdParameters copy = new AprilTagDetector.QuadThresholdParameters();
        copy.minClusterPixels = quadParams.minClusterPixels;
        copy.maxNumMaxima = quadParams.maxNumMaxima;
        copy.criticalAngle = quadParams.criticalAngle;
        copy.maxLineFitMSE = quadParams.maxLineFitMSE;
        copy.minWhiteBlackDiff = quadParams.minWhiteBlackDiff;
        copy.deglitch = quadParams.deglitch;
        return copy;
    }

    private static class RoiDecodeResult {
        final List<AprilTagDetection> detections = new ArrayList<>();
    }

    private static class ParallelWorker {
        final AprilTagDetector detector = new AprilTagDetector();
        final Mat grayRoi = new Mat();
        final Mat resizedRoi = new Mat();
        final Mat contiguousRoi = new Mat();

        void release() {
            detector.close();
            grayRoi.release();
            resizedRoi.release();
            contiguousRoi.release();
        }
    }
}
