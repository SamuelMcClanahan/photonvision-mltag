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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.CameraConfiguration;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.vision.camera.PVCameraInfo;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameProvider;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.ImageRotationMode;
import org.photonvision.vision.pipe.impl.HSVPipe;
import org.photonvision.vision.pipeline.CVPipeline;
import org.photonvision.vision.pipeline.CVPipelineSettings;
import org.photonvision.vision.pipeline.StreamDemand;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.wpilib.vision.camera.VideoMode;

/** Frames that no pipeline uses must be released, because they can hold native camera buffers. */
public class VisionRunnerFrameReleaseTest {
    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    /** Gives frames whose two images count their release calls. */
    private static class CountingProvider extends FrameProvider {
        final AtomicInteger framesGiven = new AtomicInteger();
        final AtomicInteger imagesReleased = new AtomicInteger();

        @Override
        public Frame get() {
            try {
                Thread.sleep(2);
            } catch (InterruptedException e) {
                // Keep the interrupt for the loop of VisionRunner. The frame is still given.
                Thread.currentThread().interrupt();
            }
            framesGiven.incrementAndGet();
            CVMat color =
                    new CVMat(new Mat(4, 4, CvType.CV_8UC1), imagesReleased::incrementAndGet, true);
            CVMat processed =
                    new CVMat(new Mat(4, 4, CvType.CV_8UC1), imagesReleased::incrementAndGet, true);
            return new Frame(
                    framesGiven.get(),
                    color,
                    processed,
                    FrameThresholdType.NONE,
                    new FrameStaticProperties(4, 4, 70, null));
        }

        @Override
        protected boolean checkCameraConnected() {
            return true;
        }

        @Override
        public String getName() {
            return "CountingProvider";
        }

        @Override
        public void requestFrameThresholdType(FrameThresholdType type) {}

        @Override
        public void requestFrameRotation(ImageRotationMode rotationMode) {}

        @Override
        public void requestFrameCopies(boolean copyInput, boolean copyOutput) {}

        @Override
        public void requestHsvSettings(HSVPipe.HSVParams params) {}

        @Override
        public void requestBlockForFrames(boolean blockForFrames) {}

        @Override
        public void release() {}
    }

    /** Counts the frames that it gets. */
    private static class CountingPipeline extends CVPipeline<CVPipelineResult, CVPipelineSettings> {
        final AtomicInteger framesProcessed = new AtomicInteger();

        CountingPipeline() {
            super(FrameThresholdType.NONE);
            settings = new CVPipelineSettings();
        }

        @Override
        protected void setPipeParamsImpl() {}

        @Override
        protected CVPipelineResult process(Frame frame, CVPipelineSettings settings) {
            framesProcessed.incrementAndGet();
            frame.release();
            return new CVPipelineResult(0, 0, 0, null, new Frame());
        }
    }

    private static class TestSource extends VisionSource {
        private final FrameProvider provider;

        TestSource(FrameProvider provider, CameraConfiguration configuration) {
            super(configuration);
            this.provider = provider;
            if (getCameraConfiguration().cameraQuirks == null) {
                getCameraConfiguration().cameraQuirks = QuirkyCamera.DefaultCamera;
            }
        }

        @Override
        public FrameProvider getFrameProvider() {
            return provider;
        }

        @Override
        public VisionSourceSettables getSettables() {
            return new TestSettables(getCameraConfiguration());
        }

        @Override
        public boolean isVendorCamera() {
            return false;
        }

        @Override
        public boolean hasLEDs() {
            return false;
        }

        @Override
        public void remakeSettables() {}

        @Override
        public void release() {}
    }

    private static class TestSettables extends VisionSourceSettables {
        TestSettables(CameraConfiguration configuration) {
            super(configuration);
        }

        @Override
        public void setExposureRaw(double exposure) {}

        @Override
        public void setBrightness(int brightness) {}

        @Override
        public void setGain(int gain) {}

        @Override
        public VideoMode getCurrentVideoMode() {
            return new VideoMode(0, 320, 240, 254);
        }

        @Override
        public void setVideoModeInternal(VideoMode videoMode) {}

        @Override
        public HashMap<Integer, VideoMode> getAllVideoModes() {
            var modes = new HashMap<Integer, VideoMode>();
            modes.put(0, getCurrentVideoMode());
            return modes;
        }

        @Override
        public void setAutoExposure(boolean cameraAutoExposure) {}

        @Override
        public double getMinExposureRaw() {
            return 1;
        }

        @Override
        public double getMaxExposureRaw() {
            return 1234;
        }

        @Override
        public void setAutoWhiteBalance(boolean autowb) {}

        @Override
        public void setWhiteBalanceTemp(double temp) {}

        @Override
        public double getMaxWhiteBalanceTemp() {
            return 1;
        }

        @Override
        public double getMinWhiteBalanceTemp() {
            return 2;
        }
    }

    @Test
    @SuppressWarnings("rawtypes")
    public void frameIsReleasedWhenThePipelineChangesDuringTheFrameGet() throws Exception {
        var provider = new CountingProvider();
        var conf = new CameraConfiguration(PVCameraInfo.fromFileInfo("ReleaseTest", "ReleaseTest"));
        var source = new TestSource(provider, conf);
        // The change subscriber needs a module. The module's own runner is not started.
        var module = new VisionModule(new PipelineManager(conf), source);

        var p1 = new CountingPipeline();
        var p2 = new CountingPipeline();
        // The first call of each loop gives P1 and the second call gives P2, as if the robot code
        // changed the pipeline while the vision thread waited for the frame.
        AtomicInteger calls = new AtomicInteger();
        Supplier<CVPipeline> pipelineSupplier =
                () -> calls.getAndIncrement() % 2 == 0 ? p1 : p2;

        var runner =
                new VisionRunner(
                        provider,
                        pipelineSupplier,
                        result -> {},
                        QuirkyCamera.DefaultCamera,
                        module.getChangeSubscriber(),
                        () -> 0,
                        () -> true,
                        () -> StreamDemand.NONE);
        runner.startProcess();
        long deadline = System.currentTimeMillis() + 5000;
        while (provider.framesGiven.get() < 20 && System.currentTimeMillis() < deadline) {
            Thread.sleep(10);
        }
        runner.stopProcess();

        int frames = provider.framesGiven.get();
        assertTrue(frames >= 20, "The runner took only " + frames + " frames");
        assertEquals(0, p1.framesProcessed.get() + p2.framesProcessed.get());
        // Each frame has two images and each image must be released one time.
        assertEquals(2 * frames, provider.imagesReleased.get());
    }
}
