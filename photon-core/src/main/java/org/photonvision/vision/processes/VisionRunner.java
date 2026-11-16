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

package org.photonvision.vision.processes;

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Future;
import java.util.function.Consumer;
import java.util.function.Supplier;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.dataflow.DataChangeService;
import org.photonvision.common.dataflow.events.OutgoingUIEvent;
import org.photonvision.common.dataflow.websocket.UIPhotonConfiguration;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameProvider;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.pipe.impl.HSVPipe;
import org.photonvision.vision.pipeline.AdvancedPipelineSettings;
import org.photonvision.vision.pipeline.AprilTagPipeline;
import org.photonvision.vision.pipeline.AprilTagPipelineSettings;
import org.photonvision.vision.pipeline.CVPipeline;
import org.photonvision.vision.pipeline.StreamDemand;
import org.photonvision.vision.pipeline.result.CVPipelineResult;

/** VisionRunner has a frame supplier, a pipeline supplier, and a result consumer */
@SuppressWarnings("rawtypes")
public class VisionRunner {
    private final Logger logger;
    private final Thread visionProcessThread;
    private final FrameProvider frameSupplier;
    private final Supplier<CVPipeline> pipelineSupplier;
    private final Consumer<CVPipelineResult> pipelineResultConsumer;
    private final VisionModuleChangeSubscriber changeSubscriber;
    private final List<Runnable> runnableList = new ArrayList<Runnable>();
    private final QuirkyCamera cameraQuirks;
    private final Supplier<Integer> fpsLimitSupplier;
    private final Supplier<Boolean> enabledSupplier;
    private final Supplier<StreamDemand> streamDemandSupplier;

    private long loopCount;
    private static final int TIMING_LOG_EVERY =
            EnvUtils.nonNegativeInteger("PV_VISION_TIMING_LOG_EVERY", 0);
    private long timingFrames = 0;
    private long timingSetupNanos = 0;
    private long timingFrameGetNanos = 0;
    private long timingPipelineRunNanos = 0;
    private long timingResultConsumerNanos = 0;
    private long timingCaptureToPipelineDoneNanos = 0;
    private long timingCaptureToConsumerDoneNanos = 0;
    private long timingGStreamerEstimatedAgeNanos = 0;
    private long timingGStreamerEstimatedAgeSamples = 0;

    /**
     * VisionRunner contains a thread to run a pipeline, given a frame, and will give the result to
     * the consumer.
     *
     * @param frameSupplier
     * @param pipelineSupplier
     * @param pipelineResultConsumer
     * @param cameraQuirks
     * @param changeSubscriber The subscriber to setting changes for this VisionRunner, so it can
     *     update its settings when they change.
     * @param fpsLimitSupplier
     * @param enabledSupplier
     */
    public VisionRunner(
            FrameProvider frameSupplier,
            Supplier<CVPipeline> pipelineSupplier,
            Consumer<CVPipelineResult> pipelineResultConsumer,
            QuirkyCamera cameraQuirks,
            VisionModuleChangeSubscriber changeSubscriber,
            Supplier<Integer> fpsLimitSupplier,
            Supplier<Boolean> enabledSupplier,
            Supplier<StreamDemand> streamDemandSupplier) {
        this.frameSupplier = frameSupplier;
        this.pipelineSupplier = pipelineSupplier;
        this.pipelineResultConsumer = pipelineResultConsumer;
        this.cameraQuirks = cameraQuirks;
        this.changeSubscriber = changeSubscriber;
        this.fpsLimitSupplier = fpsLimitSupplier;
        this.enabledSupplier = enabledSupplier;
        this.streamDemandSupplier = streamDemandSupplier;

        visionProcessThread = new Thread(this::update);
        visionProcessThread.setName("VisionRunner - " + frameSupplier.getName());
        logger = new Logger(VisionRunner.class, frameSupplier.getName(), LogGroup.VisionModule);
        changeSubscriber.processSettingChanges();
    }

    public void startProcess() {
        visionProcessThread.start();
    }

    public void stopProcess() {
        try {
            System.out.println("Interrupting vision process thread");
            visionProcessThread.interrupt();
            visionProcessThread.join();
        } catch (InterruptedException e) {
            logger.error("Exception killing process thread", e);
        }
    }

    public Future<Void> runSynchronously(Runnable runnable) {
        CompletableFuture<Void> future = new CompletableFuture<>();

        synchronized (runnableList) {
            runnableList.add(
                    () -> {
                        try {
                            runnable.run();
                            future.complete(null);
                        } catch (Exception ex) {
                            future.completeExceptionally(ex);
                        }
                    });
        }
        return future;
    }

    public <T> Future<T> runSynchronously(Callable<T> callable) {
        CompletableFuture<T> future = new CompletableFuture<>();

        synchronized (runnableList) {
            runnableList.add(
                    () -> {
                        try {
                            T result = callable.call();
                            future.complete(result);
                        } catch (Exception ex) {
                            future.completeExceptionally(ex);
                        }
                    });
        }

        return future;
    }

    /**
     * Waits until the next time this VisionRunner should run its pipeline, based on current FPS limit
     */
    private void waitUntilNextTick(long start) {
        int fpsLimit = fpsLimitSupplier.get();

        if (fpsLimit > 0) {
            long sleepTime = (long) (1000 / fpsLimit - (System.currentTimeMillis() - start));

            if (sleepTime > 0) {
                try {
                    Thread.sleep(sleepTime);
                } catch (InterruptedException e) {
                }
            }
            return;
        } else {
            // Fall through to no limit
            return;
        }
    }

    private void update() {
        // wait for the camera to connect
        while (!frameSupplier.isConnected() && !Thread.interrupted()) {
            // yield
            pipelineResultConsumer.accept(new CVPipelineResult(0l, 0, 0, null, new Frame()));
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                return;
            }
        }

        DataChangeService.getInstance()
                .publishEvent(
                        new OutgoingUIEvent<>(
                                "fullsettings",
                                UIPhotonConfiguration.programStateToUi(ConfigManager.getInstance().getConfig())));

        while (!Thread.interrupted()) {
            long start = System.currentTimeMillis();
            changeSubscriber.processSettingChanges();
            synchronized (runnableList) {
                for (var runnable : runnableList) {
                    try {
                        runnable.run();
                    } catch (Exception ex) {
                        logger.error("Exception running runnable", ex);
                    }
                }
                runnableList.clear();
            }

            var pipeline = pipelineSupplier.get();

            // Tell our camera implementation here what kind of pre-processing we need it to
            // be doing
            // (pipeline-dependent). I kinda hate how much leak this has...
            // TODO would a callback object be a better fit?
            long setupStartNanos = System.nanoTime();
            var settings = pipeline.getSettings();
            var streamDemand = streamDemandSupplier.get();
            var wantedProcessType = pipeline.getThresholdType();
            if (wantedProcessType == FrameThresholdType.NONE
                    && settings instanceof AprilTagPipelineSettings aprilTagSettings
                    && aprilTagSettings.useMLDetection
                    && streamDemand.effectiveOutput(settings)) {
                wantedProcessType = FrameThresholdType.GREYSCALE;
            }

            frameSupplier.requestFrameThresholdType(wantedProcessType);
            if (wantedProcessType == FrameThresholdType.HSV
                    && settings instanceof AdvancedPipelineSettings advanced) {
                var hsvParams =
                        new HSVPipe.HSVParams(
                                advanced.hsvHue, advanced.hsvSaturation, advanced.hsvValue, advanced.hueInverted);
                // TODO who should deal with preventing this from happening _every single loop_?
                frameSupplier.requestHsvSettings(hsvParams);
            }
            frameSupplier.requestFrameRotation(settings.inputImageRotationMode);
            frameSupplier.requestFrameCopies(
                    streamDemand.effectiveInput(settings), streamDemand.effectiveOutput(settings));
            frameSupplier.requestBlockForFrames(settings.blockForFrames);
            // Only the AprilTag pipeline is known to accept a 1-channel input image.
            frameSupplier.requestMonochromeInput(pipeline instanceof AprilTagPipeline);
            long setupNanos = System.nanoTime() - setupStartNanos;

            // Grab the new camera frame
            long frameGetStartNanos = System.nanoTime();
            var frame = frameSupplier.get();
            long frameGetNanos = System.nanoTime() - frameGetStartNanos;

            // Frame empty -- no point in trying to do anything more?
            if (frame.processedImage.getMat().empty() && frame.colorImage.getMat().empty()) {
                // give up without increasing loop count
                // Still feed with blank frames just dont run any pipelines

                frame.release();
                pipelineResultConsumer.accept(new CVPipelineResult(0l, 0, 0, null, new Frame()));
            } else if (pipeline == pipelineSupplier.get()) {
                if (!enabledSupplier.get()) {
                    // If we are skipping processing due to the camera being disabled, we still want to send a
                    // result with the new frame and settings, just with a null pipeline result
                    pipelineResultConsumer.accept(new CVPipelineResult(0l, 0, 0, null, new Frame()));
                    frame.release();
                    continue;
                }

                // If the pipeline has changed while we are getting our frame we should scrap
                // that frame it may result in incorrect frame settings like hsv values

                // There's no guarantee the processing type change will occur this tick, so
                // pipelines should check themselves

                // If we have an FPS limit, check if it's 0, in which case we skip processing and just send
                // a blank frame, otherwise we sleep until the next tick
                waitUntilNextTick(start);
                try {
                    long pipelineRunStartNanos = System.nanoTime();
                    var pipelineResult = pipeline.run(frame, cameraQuirks);
                    long pipelineRunNanos = System.nanoTime() - pipelineRunStartNanos;
                    if (pipelineResult == null) {
                        // The pipeline keeps this frame and gives its result with a later call.
                        loopCount++;
                        continue;
                    }
                    long pipelineDoneNanos = MathUtils.wpiNanoTime();
                    // The result can be for an earlier frame than the frame of this loop.
                    Frame resultFrame =
                            pipelineResult.inputAndOutputFrame != null
                                    ? pipelineResult.inputAndOutputFrame
                                    : frame;
                    long resultFrameTimestampNanos = resultFrame.timestampNanos;
                    long resultEstimatedAgeNanos =
                            resultFrame.captureMetadata != null
                                    ? resultFrame.captureMetadata.estimatedAgeNanos()
                                    : -1;
                    long resultConsumerStartNanos = System.nanoTime();
                    pipelineResultConsumer.accept(pipelineResult);
                    long resultConsumerNanos = System.nanoTime() - resultConsumerStartNanos;
                    long consumerDoneNanos = MathUtils.wpiNanoTime();
                    recordTiming(
                            setupNanos,
                            frameGetNanos,
                            pipelineRunNanos,
                            resultConsumerNanos,
                            resultFrameTimestampNanos,
                            resultEstimatedAgeNanos,
                            pipelineDoneNanos,
                            consumerDoneNanos);
                } catch (Exception ex) {
                    logger.error("Exception on loop " + loopCount, ex);
                }
                loopCount++;
            }
        }
    }

    private void recordTiming(
            long setupNanos,
            long frameGetNanos,
            long pipelineRunNanos,
            long resultConsumerNanos,
            long frameTimestampNanos,
            long frameEstimatedAgeNanos,
            long pipelineDoneNanos,
            long consumerDoneNanos) {
        if (TIMING_LOG_EVERY <= 0) {
            return;
        }

        timingFrames++;
        timingSetupNanos += setupNanos;
        timingFrameGetNanos += frameGetNanos;
        timingPipelineRunNanos += pipelineRunNanos;
        timingResultConsumerNanos += resultConsumerNanos;
        if (frameTimestampNanos > 0) {
            timingCaptureToPipelineDoneNanos += Math.max(0, pipelineDoneNanos - frameTimestampNanos);
            timingCaptureToConsumerDoneNanos += Math.max(0, consumerDoneNanos - frameTimestampNanos);
        }
        if (frameEstimatedAgeNanos >= 0) {
            timingGStreamerEstimatedAgeNanos += frameEstimatedAgeNanos;
            timingGStreamerEstimatedAgeSamples++;
        }

        if (timingFrames >= TIMING_LOG_EVERY) {
            logger.info(
                    String.format(
                            Locale.ROOT,
                            "Vision timing frames=%d setupMs=%.2f frameGetMs=%.2f pipelineMs=%.2f resultConsumerMs=%.2f captureToPipelineDoneMs=%.2f captureToConsumerDoneMs=%.2f gstEstimatedAgeMs=%.2f gstAgeSamples=%d",
                            timingFrames,
                            timingSetupNanos / 1_000_000.0 / timingFrames,
                            timingFrameGetNanos / 1_000_000.0 / timingFrames,
                            timingPipelineRunNanos / 1_000_000.0 / timingFrames,
                            timingResultConsumerNanos / 1_000_000.0 / timingFrames,
                            timingCaptureToPipelineDoneNanos / 1_000_000.0 / timingFrames,
                            timingCaptureToConsumerDoneNanos / 1_000_000.0 / timingFrames,
                            timingGStreamerEstimatedAgeSamples > 0
                                    ? timingGStreamerEstimatedAgeNanos
                                            / 1_000_000.0
                                            / timingGStreamerEstimatedAgeSamples
                                    : -1.0,
                            timingGStreamerEstimatedAgeSamples));
            timingFrames = 0;
            timingSetupNanos = 0;
            timingFrameGetNanos = 0;
            timingPipelineRunNanos = 0;
            timingResultConsumerNanos = 0;
            timingCaptureToPipelineDoneNanos = 0;
            timingCaptureToConsumerDoneNanos = 0;
            timingGStreamerEstimatedAgeNanos = 0;
            timingGStreamerEstimatedAgeSamples = 0;
        }
    }
}
