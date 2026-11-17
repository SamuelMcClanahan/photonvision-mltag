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

package org.photonvision.vision.frame.provider;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.CameraConfiguration;
import org.photonvision.gstreamer.GStreamerCapture;
import org.photonvision.gstreamer.GStreamerCaptureJNI;
import org.photonvision.vision.camera.PVCameraInfo;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.processes.VisionSourceSettables;
import org.wpilib.vision.camera.VideoMode;

/**
 * Tests of the open and reopen logic of the GStreamer USB provider. The tests that open a pipeline
 * need the native library (set PV_GSTREAMER_CAPTURE_JNI_LIB) and use software sources.
 */
public class GStreamerUSBFrameProviderTest {
    private static final String CAPS = "video/x-raw,format=GRAY8,width=64,height=48,framerate=30/1";
    private static final String APPSINK = "appsink name=pvappsink";

    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
    }

    private static void assumeElements(String... elements) {
        assumeTrue(GStreamerCapture.isAvailable(), "Native GStreamer capture library is not loaded");
        for (String element : elements) {
            assumeTrue(
                    GStreamerCaptureJNI.hasElement(element), "GStreamer element is not installed: " + element);
        }
    }

    private static class TestSettables extends VisionSourceSettables {
        TestSettables(String name) {
            super(new CameraConfiguration(PVCameraInfo.fromFileInfo(name, name)));
        }

        @Override
        public void setExposureRaw(double exposure) {}

        @Override
        public void setBrightness(int brightness) {}

        @Override
        public void setGain(int gain) {}

        @Override
        public VideoMode getCurrentVideoMode() {
            return new VideoMode(0, 64, 48, 30);
        }

        @Override
        protected void setVideoModeInternal(VideoMode videoMode) {}

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
            return 2;
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

    /** A provider with one fixed pipeline and settings that the test can change. */
    private static class TestProvider extends GStreamerUSBFrameProvider {
        final String pipeline;
        volatile boolean libraryAvailable = true;
        volatile long stallReopenNanos = 2_000_000_000L;
        volatile long reopenCooldownMs = 5000;

        TestProvider(String name, String pipeline) {
            super(new TestSettables(name), () -> {});
            this.pipeline = pipeline;
            requestFrameThresholdType(FrameThresholdType.NONE);
            requestFrameCopies(false, false);
        }

        @Override
        int[] ioModeCandidates() {
            return new int[] {0};
        }

        @Override
        List<PipelineCandidate> pipelineCandidates(int ioMode) {
            return List.of(
                    new PipelineCandidate(pipeline, GStreamerCaptureJNI.OUTPUT_AUTO, "test pipeline"));
        }

        @Override
        boolean nativeLibraryAvailable() {
            return libraryAvailable;
        }

        @Override
        long stallReopenNanos() {
            return stallReopenNanos;
        }

        @Override
        long reopenCooldownMs() {
            return reopenCooldownMs;
        }
    }

    /** Returns true and releases the frame if it has an image. */
    private static boolean hasImage(Frame frame) {
        boolean image = !frame.colorImage.getMat().empty();
        frame.release();
        return image;
    }

    @Test
    public void silentCameraDoesNotBlockTheOpenOfAnotherCamera() throws Exception {
        assumeElements("appsrc", "videotestsrc");
        // A reaches PLAYING as a live source but never gives a buffer. With async=false the appsink
        // does not wait for a first buffer in the state change, so the native create returns at
        // once, as for a camera that fails after PLAYING. Without it the native create itself
        // waits up to 3 s for the preroll of the appsink.
        var silent =
                new TestProvider(
                        "Silent", "appsrc is-live=true ! " + CAPS + " ! " + APPSINK + " async=false");
        var good =
                new TestProvider("Good", "videotestsrc is-live=true ! " + CAPS + " ! " + APPSINK);

        Thread silentThread = new Thread(() -> hasImage(silent.get()));
        silentThread.start();
        try {
            Thread.sleep(300);

            long start = System.nanoTime();
            boolean gotFrame = false;
            while (!gotFrame && System.nanoTime() - start < 10_000_000_000L) {
                gotFrame = hasImage(good.get());
            }
            long elapsedMs = (System.nanoTime() - start) / 1_000_000L;
            assertTrue(gotFrame, "The good camera gave no frame");
            assertTrue(
                    elapsedMs < 1500,
                    "The good camera waited " + elapsedMs + " ms for the open of the silent camera");
        } finally {
            silentThread.join(20_000);
            silent.release();
            good.release();
        }
    }

    @Test
    public void endedStreamDoesNotSpinAndIsOpenedAgain() throws Exception {
        assumeElements("videotestsrc");
        var provider =
                new TestProvider(
                        "Ending",
                        "videotestsrc is-live=true num-buffers=15 ! " + CAPS + " ! " + APPSINK);
        try {
            // Read frames until the stream ends.
            int frames = 0;
            long deadline = System.nanoTime() + 10_000_000_000L;
            boolean ended = false;
            while (System.nanoTime() < deadline) {
                if (hasImage(provider.get())) {
                    frames++;
                } else if (frames > 0) {
                    ended = true;
                    break;
                }
            }
            assertTrue(ended, "The stream did not end; frames=" + frames);

            // After the end the vision loop must not spin.
            int calls = 0;
            long spinEnd = System.nanoTime() + 500_000_000L;
            while (System.nanoTime() < spinEnd) {
                provider.get().release();
                calls++;
            }
            assertTrue(calls < 40, "get() returned " + calls + " times in 500 ms after the end");

            // With a short stall time and no cooldown the camera opens again.
            provider.stallReopenNanos = 100_000_000L;
            provider.reopenCooldownMs = 0;
            boolean reopened = false;
            deadline = System.nanoTime() + 10_000_000_000L;
            while (!reopened && System.nanoTime() < deadline) {
                reopened = hasImage(provider.get());
            }
            assertTrue(reopened, "The camera was not opened again after the stream ended");
        } finally {
            provider.release();
        }
    }

    @Test
    public void missingLibraryFailsAtOnceWithTheCause() {
        var provider = new TestProvider("NoLibrary", "videotestsrc ! " + CAPS + " ! " + APPSINK);
        provider.libraryAvailable = false;

        PrintStream originalOut = System.out;
        var captured = new ByteArrayOutputStream();
        long elapsedMs;
        Frame frame;
        System.setOut(new PrintStream(captured, true, StandardCharsets.UTF_8));
        try {
            long start = System.nanoTime();
            frame = provider.get();
            elapsedMs = (System.nanoTime() - start) / 1_000_000L;
        } finally {
            System.setOut(originalOut);
        }
        String log = captured.toString(StandardCharsets.UTF_8);
        originalOut.print(log);

        assertFalse(hasImage(frame));
        // get() sleeps 100 ms after a failed open. No candidate is tried.
        assertTrue(elapsedMs < 1000, "get() took " + elapsedMs + " ms");
        assertTrue(
                log.contains("Native GStreamer capture library is not loaded"),
                "The log has no library message: " + log);
        assertTrue(log.contains(GStreamerCaptureJNI.JNI_LIB_ENV), "The log has no library path");
        assertFalse(provider.isConnected(), "The provider must report the failed open");
        provider.release();
    }
}
