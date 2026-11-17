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

package org.photonvision.vision.pipeline;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.core.RotatedRect;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.configuration.NeuralNetworkModelManager;
import org.photonvision.common.dataflow.structures.Packet;
import org.photonvision.common.hardware.Platform;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.estimation.TargetModel;
import org.photonvision.targeting.MultiTargetPNPResult;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.objects.Model;
import org.photonvision.vision.pipe.CVPipe.CVPipeResult;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe;
import org.photonvision.vision.pipe.impl.AprilTagDetectionPipe.AprilTagDetectionPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagMLHybridPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe;
import org.photonvision.vision.pipe.impl.AprilTagPoseEstimatorPipe.AprilTagPoseEstimatorPipeParams;
import org.photonvision.vision.pipe.impl.AprilTagROIDecodePipe;
import org.photonvision.vision.pipe.impl.AprilTagROIDetectionPipe;
import org.photonvision.vision.pipe.impl.CalculateFPSPipe;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe;
import org.photonvision.vision.pipe.impl.MultiTargetPNPPipe.MultiTargetPNPPipeParams;
import org.photonvision.vision.pipe.impl.VisionWorkerPool;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TrackedTarget;
import org.photonvision.vision.target.TrackedTarget.TargetCalculationParameters;
import org.wpilib.math.geometry.CoordinateSystem;
import org.wpilib.math.geometry.Pose3d;
import org.wpilib.math.geometry.Rotation3d;
import org.wpilib.math.geometry.Transform3d;
import org.wpilib.math.util.Units;
import org.wpilib.vision.apriltag.AprilTagDetection;
import org.wpilib.vision.apriltag.AprilTagDetector;
import org.wpilib.vision.apriltag.AprilTagPoseEstimate;
import org.wpilib.vision.apriltag.AprilTagPoseEstimator.Config;

public class AprilTagPipeline extends CVPipeline<CVPipelineResult, AprilTagPipelineSettings> {
    private static final Logger logger = new Logger(AprilTagPipeline.class, LogGroup.VisionModule);

    private final AprilTagDetectionPipe aprilTagDetectionPipe = new AprilTagDetectionPipe();
    private final AprilTagPoseEstimatorPipe singleTagPoseEstimatorPipe =
            new AprilTagPoseEstimatorPipe();

    /**
     * Estimate the poses of different tags on different threads. The estimate of one tag does not use
     * data of a different tag, so the poses are equal to the poses of the serial calculation. Set
     * PV_MLTAG_PARALLEL_POSE=false to estimate all poses on the vision thread.
     */
    private static final boolean PARALLEL_POSE = EnvUtils.flag("PV_MLTAG_PARALLEL_POSE", true);

    // One pose pipe for each group of tags that runs at the same time. Index 0 is
    // singleTagPoseEstimatorPipe. A pipe keeps scratch data, so two threads cannot share one.
    private final List<AprilTagPoseEstimatorPipe> posePipes = new ArrayList<>();
    private AprilTagPoseEstimatorPipeParams poseParams = null;
    private final MultiTargetPNPPipe multiTagPNPPipe = new MultiTargetPNPPipe();
    private final CalculateFPSPipe calculateFPSPipe = new CalculateFPSPipe();

    // ML-assisted detection (composite ROI detect + ROI decode)
    private final AprilTagMLHybridPipe mlHybridPipe;
    private boolean mlAvailable = false;
    private boolean mlWasAvailable = false;
    private boolean pipeParamsConfigured = false;
    private int lastPipeParamsHash = 0;

    // Frames that are in work. Only used when the hybrid pipe has more than one inference lane.
    // The detection is kept so that a frame whose work task was cancelled is released only after
    // its lane does not use it.
    private record PendingFrame(
            Frame frame,
            AprilTagMLHybridPipe.PendingDetection detection,
            Future<CVPipelineResult> result) {}

    private final ArrayDeque<PendingFrame> pendingFrames = new ArrayDeque<>();
    private InferenceOverlapController overlapController;

    /**
     * The work thread of this pipeline for the multi-lane mode. It does the work that follows the
     * model inference (tag decode, pose) for one frame after the other, in the capture order. The
     * vision thread then only starts frames and returns results, so a slow frame does not stop it
     * from getting the next camera frame.
     */
    private ExecutorService frameWorker;

    private static final AtomicInteger frameWorkerNumber = new AtomicInteger();

    // Time of the tag decode and pose work of the last frame. Written by the work thread.
    private volatile long lastPostInferenceNanos = -1;

    // The overlap depth of the last log line. -1 = no log line until now.
    private int loggedOverlapDepth = -1;

    /** A pending frame that is older than this is not completed. Its result has no use. */
    private static final long STALE_PENDING_NANOS = 500_000_000L;

    private static final FrameThresholdType PROCESSING_TYPE = FrameThresholdType.GREYSCALE;

    public AprilTagPipeline() {
        this(new AprilTagPipelineSettings());
    }

    public AprilTagPipeline(AprilTagPipelineSettings settings) {
        super(PROCESSING_TYPE);
        this.settings = settings;
        this.mlHybridPipe = new AprilTagMLHybridPipe();
    }

    /**
     * @param settings The pipeline settings.
     * @param inferenceLanes Number of frames that can be in model inference at the same time.
     */
    public AprilTagPipeline(AprilTagPipelineSettings settings, int inferenceLanes) {
        super(PROCESSING_TYPE);
        this.settings = settings;
        this.mlHybridPipe = new AprilTagMLHybridPipe(inferenceLanes);
    }

    @Override
    public FrameThresholdType getThresholdType() {
        if (settings != null && settings.useMLDetection && !settings.outputShouldShow) {
            return FrameThresholdType.NONE;
        }
        return PROCESSING_TYPE;
    }

    @Override
    protected void setPipeParamsImpl() {
        // Sanitize thread count - not supported to have fewer than 1 threads
        settings.threads = Math.max(1, settings.threads);

        int pipeParamsHash = pipeParamsHash();
        if (pipeParamsConfigured && pipeParamsHash == lastPipeParamsHash) {
            return;
        }

        // The settings changed. Frames that are in work use the old model and the old settings.
        // Wait for them and release them before the pipes get new parameters.
        discardPendingFrames();

        // for now, hard code tag width based on enum value
        // From 2024 best guess is 6.5
        double tagWidth = Units.inchesToMeters(6.5);
        TargetModel tagModel = TargetModel.kAprilTag36h11;
        if (settings.tagFamily == AprilTagFamily.kTag16h5) {
            // 2023 tag, 6in
            tagWidth = Units.inchesToMeters(6);
            tagModel = TargetModel.kAprilTag16h5;
        }

        var config = new AprilTagDetector.Config();
        config.numThreads = settings.threads;
        config.refineEdges = settings.refineEdges;
        config.quadSigma = (float) settings.blur;
        config.quadDecimate = settings.decimate;

        var quadParams = new AprilTagDetector.QuadThresholdParameters();
        // 5 was the default minClusterPixels in WPILib prior to 2025
        // increasing it causes detection problems when decimate > 1
        quadParams.minClusterPixels = 5;
        // these are the same as the values in WPILib 2025
        // setting them here to prevent upstream changes from changing behavior of the detector
        quadParams.maxNumMaxima = 10;
        quadParams.criticalAngle = 45 * Math.PI / 180.0;
        quadParams.maxLineFitMSE = 10.0f;
        quadParams.minWhiteBlackDiff = 5;
        quadParams.deglitch = false;

        aprilTagDetectionPipe.setParams(
                new AprilTagDetectionPipeParams(settings.tagFamily, config, quadParams));

        if (frameStaticProperties.cameraCalibration != null) {
            var cameraMatrix = frameStaticProperties.cameraCalibration.getCameraIntrinsicsMat();
            if (cameraMatrix != null && cameraMatrix.rows() > 0) {
                var cx = cameraMatrix.get(0, 2)[0];
                var cy = cameraMatrix.get(1, 2)[0];
                var fx = cameraMatrix.get(0, 0)[0];
                var fy = cameraMatrix.get(1, 1)[0];

                poseParams =
                        new AprilTagPoseEstimatorPipeParams(
                                new Config(tagWidth, fx, fy, cx, cy),
                                frameStaticProperties.cameraCalibration,
                                settings.numIterations);
                singleTagPoseEstimatorPipe.setParams(poseParams);
                for (AprilTagPoseEstimatorPipe posePipe : posePipes) {
                    posePipe.setParams(poseParams);
                }

                // TODO global state ew
                var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();
                multiTagPNPPipe.setParams(
                        new MultiTargetPNPPipeParams(frameStaticProperties.cameraCalibration, atfl, tagModel));
            }
        }

        // ML-assisted detection configuration
        if (settings.useMLDetection) {
            boolean platformOk = checkMLAvailability();
            Model apriltagModel = null;
            if (platformOk) {
                apriltagModel = resolveModel(settings);
            }

            if (platformOk && apriltagModel != null) {
                AprilTagROIDetectionPipe.AprilTagROIDetectionParams detectionParams =
                        new AprilTagROIDetectionPipe.AprilTagROIDetectionParams(
                                apriltagModel, settings.mlConfidenceThreshold, settings.mlNmsThreshold);

                AprilTagROIDecodePipe.ROIDecodeParams decodeParams =
                        new AprilTagROIDecodePipe.ROIDecodeParams();
                decodeParams.tagFamily = settings.tagFamily;
                decodeParams.maxHammingDistance = settings.hammingDist;
                decodeParams.minDecisionMargin = settings.decisionMargin;
                decodeParams.detectorConfig.numThreads = settings.threads;
                decodeParams.detectorConfig.refineEdges = settings.refineEdges;
                decodeParams.detectorConfig.quadDecimate = settings.decimate;
                decodeParams.detectorConfig.quadSigma = (float) settings.blur;

                // ATR (Adaptive Tag Resizing) settings
                decodeParams.atrEnabled = settings.atrEnabled;
                decodeParams.atrTargetDimension = settings.atrTargetDimension;
                decodeParams.atrMinScaleFactor = settings.atrMinScaleFactor;

                mlHybridPipe.setParams(
                        new AprilTagMLHybridPipe.Params(
                                detectionParams, decodeParams, settings.mlRoiPaddingPixels));
            }

            mlAvailable = platformOk && apriltagModel != null && mlHybridPipe.isAvailable();

            if (!mlWasAvailable && mlAvailable) {
                logger.info("ML-assisted AprilTag detection enabled");
            }
            if (platformOk && apriltagModel == null && mlWasAvailable) {
                logger.warn("ML-assisted detection enabled but no AprilTag model found");
            }
            if (!platformOk && mlWasAvailable) {
                logger.debug("ML-assisted detection not available on this platform");
            }
        } else {
            mlAvailable = false;
        }
        mlWasAvailable = mlAvailable;
        lastPipeParamsHash = pipeParamsHash;
        pipeParamsConfigured = true;
    }

    private int pipeParamsHash() {
        final int prime = 31;
        int result = settings.hashCode();
        result = prime * result + frameStaticProperties.imageWidth;
        result = prime * result + frameStaticProperties.imageHeight;
        long fovBits = Double.doubleToLongBits(frameStaticProperties.fov);
        result = prime * result + (int) (fovBits ^ (fovBits >>> 32));
        result = prime * result + System.identityHashCode(frameStaticProperties.cameraCalibration);
        return result;
    }

    @Override
    protected CVPipelineResult process(Frame frame, AprilTagPipelineSettings settings) {
        boolean useML = settings.useMLDetection && mlAvailable;

        if (!useML && frame.type != FrameThresholdType.GREYSCALE) {
            // We asked for a GREYSCALE frame, but didn't get one -- best we can do is give up
            return new CVPipelineResult(frame.sequenceID, 0, 0, List.of(), frame);
        }

        if (useML && mlHybridPipe.getLaneCount() > 1) {
            return processWithInferenceLanes(frame, settings);
        }
        if (!pendingFrames.isEmpty()) {
            discardPendingFrames();
        }

        // Perform AprilTag detection (traditional or ML-assisted)
        List<AprilTagDetection> detections;
        long detectionNanos;
        List<RotatedRect> mlDetectionRois = List.of();

        if (useML) {
            // Use ML-assisted hybrid detection
            var hybridResult = mlHybridPipe.run(frame);
            detections = hybridResult.output.detections();
            detectionNanos = hybridResult.nanosElapsed;
            mlDetectionRois = hybridResult.output.rois();
        } else {
            // Use traditional detection
            CVPipeResult<List<AprilTagDetection>> tagDetectionPipeResult =
                    aprilTagDetectionPipe.run(frame.processedImage);
            detections = tagDetectionPipeResult.output;
            detectionNanos = tagDetectionPipeResult.nanosElapsed;
        }

        return buildResult(frame, settings, detections, detectionNanos, mlDetectionRois);
    }

    /**
     * ML detection with more than one frame in work at the same time.
     *
     * <p>This method starts the work of the input frame: the model inference on a free lane, then the
     * tag decode and the pose estimation on the work thread of this pipeline. Then it returns the
     * result of the oldest frame that is in work. The results are returned in the capture order, and
     * each frame gets the same processing as in {@link #process}.
     *
     * <p>{@link InferenceOverlapController} decides if a call returns the result of its own frame
     * (when the pipeline is faster than the camera) or of an earlier frame (when the work for one
     * frame takes longer than the camera frame period).
     *
     * @return The result of the oldest frame, or null if this call gives no result.
     */
    private CVPipelineResult processWithInferenceLanes(
            Frame frame, AprilTagPipelineSettings settings) {
        if (overlapController == null) {
            overlapController =
                    new InferenceOverlapController(
                            mlHybridPipe.getLaneCount(),
                            EnvUtils.decimal(
                                    "PV_MLTAG_OVERLAP_RESERVE", InferenceOverlapController.DEFAULT_RESERVE));
        }
        if (frameWorker == null) {
            frameWorker =
                    Executors.newSingleThreadExecutor(
                            runnable -> {
                                Thread thread =
                                        new Thread(runnable, "MLTag-Frame-" + frameWorkerNumber.incrementAndGet());
                                thread.setDaemon(true);
                                return thread;
                            });
        }
        discardStalePendingFrames(frame.timestampNanos);

        var action =
                overlapController.onFrame(
                        System.nanoTime(),
                        frame.timestampNanos,
                        pendingFrames.size(),
                        mlHybridPipe.getInferenceNanosAverage(),
                        mlHybridPipe.getInferenceSamples());

        // One log line for each change of the mode, to see on the coprocessor what the pipeline does.
        int overlapDepth = overlapController.getDepth();
        if (overlapDepth != loggedOverlapDepth) {
            logger.info(
                    overlapDepth == 0
                            ? "ML AprilTag inference lanes: one frame at a time"
                            : "ML AprilTag inference lanes: " + overlapDepth + " frames in work");
            loggedOverlapDepth = overlapDepth;
        }

        if (action == InferenceOverlapController.Action.SKIP_INPUT && !pendingFrames.isEmpty()) {
            // The pipeline goes back to one frame at a time. This call must return an earlier
            // frame, so it cannot also return the input frame.
            frame.release();
        } else {
            // The inference starts now on a lane. The work thread takes the frames in the capture
            // order: it waits for the inference of a frame and then does the remaining work.
            final var detection = mlHybridPipe.submit(frame);
            final AprilTagPipelineSettings frameSettings = settings;
            pendingFrames.addLast(
                    new PendingFrame(
                            frame, detection, frameWorker.submit(() -> finishFrame(detection, frameSettings))));
            if (action == InferenceOverlapController.Action.DEFER) {
                overlapController.onCallEnd(System.nanoTime(), -1);
                return null;
            }
        }

        var pending = pendingFrames.removeFirst();
        CVPipelineResult result = awaitResult(pending);
        if (result == null) {
            // The processing of this frame failed. As in the one-lane path, it gives no result.
            pending.frame().release();
            overlapController.onCallEnd(System.nanoTime(), -1);
            return null;
        }
        overlapController.onCallEnd(System.nanoTime(), lastPostInferenceNanos);
        return result;
    }

    /**
     * Returns the result of the oldest frame in work if it is complete, else null. VisionRunner calls
     * this when the camera gives no frame or is disabled, so the results of the frames in work are
     * published and not lost when they become stale.
     */
    @Override
    public CVPipelineResult pollCompletedResult() {
        PendingFrame pending;
        while ((pending = pendingFrames.peekFirst()) != null && pending.result().isDone()) {
            pendingFrames.removeFirst();
            CVPipelineResult result = awaitResult(pending);
            if (result == null) {
                // A failed frame gives no result. Try the next frame.
                pending.frame().release();
                continue;
            }
            // CVPipeline.run() does not see this result, so set its capture time here.
            result.setImageCaptureTimestampNanos(result.inputAndOutputFrame.timestampNanos);
            return result;
        }
        return null;
    }

    @Override
    public Frame getResultSourceFrame(CVPipelineResult result, Frame inputFrame) {
        // In the multi-lane mode a result can be for an earlier frame than the input frame. Each
        // result holds the frame that it was computed from.
        return result.inputAndOutputFrame != null ? result.inputAndOutputFrame : inputFrame;
    }

    /**
     * Runs on the work thread: the work for one frame that follows the model inference.
     *
     * @return The result, or null if the processing of the frame failed. In both cases the lane does
     *     not use the frame when this method returns.
     */
    private CVPipelineResult finishFrame(
            AprilTagMLHybridPipe.PendingDetection detection, AprilTagPipelineSettings frameSettings) {
        try {
            var completed = mlHybridPipe.complete(detection);
            long buildStart = System.nanoTime();
            CVPipelineResult result =
                    buildResult(
                            detection.frame,
                            frameSettings,
                            completed.result().detections(),
                            completed.nanosElapsed(),
                            completed.result().rois());
            lastPostInferenceNanos = completed.decodeNanos() + System.nanoTime() - buildStart;
            return result;
        } catch (InterruptedException e) {
            // The work thread is stopped. The frame gives no result, but the caller can release
            // the frame only after the lane does not use it. Set the flag again after the wait.
            detection.await();
            Thread.currentThread().interrupt();
            return null;
        } catch (Throwable e) {
            // No failure can return before the lane is done with the frame.
            detection.await();
            logger.error("ML AprilTag frame processing failed", e);
            return null;
        }
    }

    /**
     * Waits for the result of a frame. An interrupt does not stop the wait, because the lane and the
     * work thread can still use the frame. The interrupt flag is set again after the wait.
     *
     * @return The result that owns the frame, or null if the processing of the frame failed or was
     *     cancelled. Then the caller must release the frame. In both cases no other thread uses the
     *     frame when this method returns.
     */
    private CVPipelineResult awaitResult(PendingFrame pending) {
        boolean interrupted = false;
        try {
            while (true) {
                try {
                    return pending.result().get();
                } catch (InterruptedException e) {
                    interrupted = true;
                } catch (CancellationException e) {
                    // The work task did not start (release() cancelled it), so it did not wait
                    // for the lane. Wait for the lane here.
                    pending.detection().await();
                    return null;
                } catch (ExecutionException e) {
                    logger.error("ML AprilTag frame processing failed", e);
                    return null;
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }

    /** Releases the frames in work if the oldest one is too old, for example after a pause. */
    private void discardStalePendingFrames(long newestTimestampNanos) {
        PendingFrame oldest = pendingFrames.peekFirst();
        if (oldest != null
                && newestTimestampNanos - oldest.frame().timestampNanos > STALE_PENDING_NANOS) {
            // Discard all frames, not only the stale ones. The newer frames are also late, and the
            // one-frame-at-a-time mode of the controller after the reset needs an empty queue.
            discardPendingFrames();
            logger.warn("ML AprilTag: frames that were in work for too long were discarded");
        }
    }

    /** Waits for all frames that are in work and releases them. */
    private void discardPendingFrames() {
        PendingFrame pending;
        while ((pending = pendingFrames.pollFirst()) != null) {
            CVPipelineResult result = awaitResult(pending);
            if (result != null) {
                result.release();
            } else {
                pending.frame().release();
            }
        }
        if (overlapController != null) {
            overlapController.reset();
        }
    }

    /**
     * Filters the detections, estimates the poses and makes the pipeline result for one frame.
     *
     * @param frame The frame that the detections are from.
     */
    private CVPipelineResult buildResult(
            Frame frame,
            AprilTagPipelineSettings settings,
            List<AprilTagDetection> detections,
            long detectionNanos,
            List<RotatedRect> mlDetectionRois) {
        long sumPipeNanosElapsed = detectionNanos;
        // In the multi-lane mode this method runs on the work thread, so it uses the properties
        // of its frame and not the field that the vision thread sets for each call.
        final FrameStaticProperties frameStaticProperties =
                frame.frameStaticProperties != null
                        ? frame.frameStaticProperties
                        : this.frameStaticProperties;

        List<AprilTagDetection> usedDetections = new ArrayList<>();
        List<TrackedTarget> targetList = new ArrayList<>();

        // Filter out detections based on pipeline settings
        for (AprilTagDetection detection : detections) {
            // TODO this should be in a pipe, not in the top level here (Matt)
            if (detection.getDecisionMargin() < settings.decisionMargin) continue;
            if (detection.getHamming() > settings.hammingDist) continue;

            usedDetections.add(detection);

            // Populate target list for multitag
            // (TODO: Address circular dependencies. Multitag only requires corners and IDs, this should
            // not be necessary.)
            TrackedTarget target =
                    new TrackedTarget(
                            detection,
                            null,
                            new TargetCalculationParameters(
                                    false, null, null, null, null, frameStaticProperties));

            targetList.add(target);
        }

        Optional<MultiTargetPNPResult> multiTagResult = Optional.empty();

        if (settings.solvePNPEnabled && settings.doMultiTarget) {
            var multiTagOutput = multiTagPNPPipe.run(targetList);
            sumPipeNanosElapsed += multiTagOutput.nanosElapsed;
            multiTagResult = multiTagOutput.output;
        }

        // Do single-tag pose estimation
        if (settings.solvePNPEnabled) {
            // Clear target list that was used for multitag so we can add target transforms
            targetList.clear();
            // TODO global state again ew
            var atfl = ConfigManager.getInstance().getConfig().getApriltagFieldLayout();

            // Do single-tag estimation when "always enabled" or if a tag was not used for multitag
            boolean[] needsSingleTagPose = new boolean[usedDetections.size()];
            for (int i = 0; i < usedDetections.size(); i++) {
                needsSingleTagPose[i] =
                        settings.doSingleTargetAlways
                                || !(multiTagResult.isPresent()
                                        && multiTagResult
                                                .get()
                                                .fiducialIDsUsed
                                                .contains((short) usedDetections.get(i).getId()));
            }
            AprilTagPoseEstimate[] singleTagPoses = new AprilTagPoseEstimate[usedDetections.size()];
            sumPipeNanosElapsed +=
                    estimateSingleTagPoses(usedDetections, needsSingleTagPose, singleTagPoses);

            for (int detectionIndex = 0; detectionIndex < usedDetections.size(); detectionIndex++) {
                AprilTagDetection detection = usedDetections.get(detectionIndex);
                AprilTagPoseEstimate tagPoseEstimate = singleTagPoses[detectionIndex];

                // If single-tag estimation was not done, this tag was used in multi-tag estimation
                if (tagPoseEstimate == null && multiTagResult.isPresent()) {
                    // compute this tag's camera-to-tag transform using the multitag result
                    var tagPose = atfl.getTagPose(detection.getId());
                    if (tagPose.isPresent()) {
                        var camToTag =
                                new Transform3d(
                                        new Pose3d().plus(multiTagResult.get().estimatedPose.best), tagPose.get());
                        // match expected AprilTag coordinate system
                        camToTag =
                                CoordinateSystem.convert(camToTag, CoordinateSystem.NWU(), CoordinateSystem.EDN());
                        // (AprilTag expects Z axis going into tag)
                        camToTag =
                                new Transform3d(
                                        camToTag.getTranslation(),
                                        new Rotation3d(0, Math.PI, 0).rotateBy(camToTag.getRotation()));
                        tagPoseEstimate = new AprilTagPoseEstimate(camToTag, camToTag, 0, 0);
                    }
                }

                // populate the target list
                // Challenge here is that TrackedTarget functions with OpenCV Contour
                TrackedTarget target =
                        new TrackedTarget(
                                detection,
                                tagPoseEstimate,
                                new TargetCalculationParameters(
                                        false, null, null, null, null, frameStaticProperties));

                var correctedBestPose =
                        MathUtils.convertOpenCVtoPhotonTransform(target.getBestCameraToTarget3d());
                var correctedAltPose =
                        MathUtils.convertOpenCVtoPhotonTransform(target.getAltCameraToTarget3d());

                target.setBestCameraToTarget3d(
                        new Transform3d(correctedBestPose.getTranslation(), correctedBestPose.getRotation()));
                target.setAltCameraToTarget3d(
                        new Transform3d(correctedAltPose.getTranslation(), correctedAltPose.getRotation()));

                targetList.add(target);
            }
        }

        if (targetList.size() > Packet.MAX_ARRAY_LEN) {
            logger.error(
                    "We have " + targetList.size() + " targets! Arbitrarily dropping some on the floor");
            targetList = targetList.subList(0, Packet.MAX_ARRAY_LEN);
        }

        var fpsResult = calculateFPSPipe.run(null);
        var fps = fpsResult.output;

        return new CVPipelineResult(
                frame.sequenceID,
                sumPipeNanosElapsed,
                fps,
                targetList,
                multiTagResult,
                frame,
                List.of(),
                mlDetectionRois);
    }

    /**
     * Estimates the single-tag poses. With two or more tags, the tags are put into groups and the
     * groups run at the same time on the calling thread and on the shared worker pool. The result for
     * each tag is the same as the result of a serial calculation.
     *
     * @param detections The detections of the frame.
     * @param needsPose Selects the detections that get a pose.
     * @param poses Output: the pose of each selected detection, at the index of the detection.
     * @return The sum of the calculation time of all tags, in nanoseconds.
     */
    private long estimateSingleTagPoses(
            List<AprilTagDetection> detections, boolean[] needsPose, AprilTagPoseEstimate[] poses) {
        List<Integer> selected = new ArrayList<>(detections.size());
        for (int i = 0; i < detections.size(); i++) {
            if (needsPose[i]) {
                selected.add(i);
            }
        }
        if (selected.isEmpty()) {
            return 0;
        }

        int groupCount =
                PARALLEL_POSE && poseParams != null
                        ? Math.min(selected.size(), VisionWorkerPool.maxGroups(settings.threads))
                        : 1;
        if (groupCount <= 1) {
            long nanos = 0;
            for (int index : selected) {
                var poseResult = singleTagPoseEstimatorPipe.run(detections.get(index));
                nanos += poseResult.nanosElapsed;
                poses[index] = poseResult.output;
            }
            return nanos;
        }

        // Create the calibration matrices on this thread before other threads read them. Use the
        // calibration that the pose pipes read: the frame can have no calibration while poseParams
        // is still set from an earlier calibration.
        poseParams.calibration().getCameraIntrinsicsMat();
        poseParams.calibration().getDistCoeffsMat();

        if (posePipes.isEmpty()) {
            posePipes.add(singleTagPoseEstimatorPipe);
        }
        while (posePipes.size() < groupCount) {
            var posePipe = new AprilTagPoseEstimatorPipe();
            posePipe.setParams(poseParams);
            posePipes.add(posePipe);
        }

        long[] groupNanos = new long[groupCount];
        List<Future<?>> futures = new ArrayList<>(groupCount - 1);
        for (int group = 1; group < groupCount; group++) {
            final int groupIndex = group;
            futures.add(
                    VisionWorkerPool.EXECUTOR.submit(
                            () ->
                                    estimatePoseGroup(
                                            detections, selected, groupIndex, groupCount, poses, groupNanos)));
        }
        estimatePoseGroup(detections, selected, 0, groupCount, poses, groupNanos);

        // Wait for all tasks also when this thread is interrupted: the tasks write into poses and
        // use the pose pipes. Set the interrupt flag again after the wait.
        boolean failed = false;
        boolean interrupted = false;
        for (Future<?> future : futures) {
            while (true) {
                try {
                    future.get();
                    break;
                } catch (InterruptedException e) {
                    interrupted = true;
                } catch (ExecutionException | CancellationException e) {
                    logger.error("Parallel pose estimation failed", e);
                    failed = true;
                    break;
                }
            }
        }
        if (interrupted) {
            Thread.currentThread().interrupt();
        }
        if (failed) {
            // All tasks are complete here. Calculate the poses that are missing on this thread.
            for (int index : selected) {
                if (poses[index] == null) {
                    poses[index] = singleTagPoseEstimatorPipe.run(detections.get(index)).output;
                }
            }
        }

        long nanos = 0;
        for (long value : groupNanos) {
            nanos += value;
        }
        return nanos;
    }

    private void estimatePoseGroup(
            List<AprilTagDetection> detections,
            List<Integer> selected,
            int group,
            int groupCount,
            AprilTagPoseEstimate[] poses,
            long[] groupNanos) {
        AprilTagPoseEstimatorPipe posePipe = posePipes.get(group);
        long nanos = 0;
        for (int i = group; i < selected.size(); i += groupCount) {
            int index = selected.get(i);
            var poseResult = posePipe.run(detections.get(index));
            nanos += poseResult.nanosElapsed;
            poses[index] = poseResult.output;
        }
        groupNanos[group] = nanos;
    }

    /**
     * Checks if ML detection is available on the current platform. Currently supported: RK3588
     * (Orange Pi 5, Rock 5C, CoolPi 4B), QCS6490 (Rubik Pi 3), and NVIDIA Jetson via TensorRT.
     */
    protected boolean checkMLAvailability() {
        Platform platform = Platform.getCurrentPlatform();
        return platform == Platform.LINUX_QCS6490
                || platform == Platform.LINUX_RK3588_64
                || Platform.isJetson();
    }

    /** Returns the model that the settings select, or null if it is not available. */
    protected Model resolveModel(AprilTagPipelineSettings settings) {
        if (settings.model == null) {
            return null;
        }
        return NeuralNetworkModelManager.getInstance()
                .getModel(settings.model.modelPath().toString())
                .orElse(null);
    }

    @Override
    public void release() {
        // VisionModule.stop() and a pipeline change can both release this pipeline. A second call
        // does nothing, so no pipe is released two times.
        if (released) {
            return;
        }
        // A call of run() that starts after this point fails at once and does not use the pipes.
        released = true;
        discardPendingFrames();
        if (frameWorker != null) {
            // Cancel the tasks that did not start, so a thread that waits for one of them does not
            // wait without end. Do not set frameWorker to null: a call of run() that is still in
            // work must not make a new work thread.
            for (Runnable task : frameWorker.shutdownNow()) {
                if (task instanceof Future<?> future) {
                    future.cancel(false);
                }
            }
            // The work thread uses the decode and pose pipes. Wait for it before they are released.
            awaitTermination(frameWorker);
        }
        aprilTagDetectionPipe.release();
        singleTagPoseEstimatorPipe.release();
        for (AprilTagPoseEstimatorPipe posePipe : posePipes) {
            if (posePipe != singleTagPoseEstimatorPipe) {
                posePipe.release();
            }
        }
        posePipes.clear();
        mlHybridPipe.release();
        super.release();
    }

    /** Waits up to 2 s for the end of the threads of an executor. An interrupt does not stop it. */
    private static void awaitTermination(ExecutorService executor) {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
        boolean interrupted = false;
        try {
            while (true) {
                long remaining = deadline - System.nanoTime();
                try {
                    if (executor.awaitTermination(Math.max(0, remaining), TimeUnit.NANOSECONDS)) {
                        return;
                    }
                    if (remaining <= 0) {
                        logger.warn("ML AprilTag: the frame work thread did not end in 2 s");
                        return;
                    }
                } catch (InterruptedException e) {
                    interrupted = true;
                }
            }
        } finally {
            if (interrupted) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
