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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;
import org.photonvision.gstreamer.GStreamerCaptureJNI;
import org.photonvision.vision.frame.provider.GStreamerUSBFrameProvider.GStreamerBackend;
import org.photonvision.vision.frame.provider.GStreamerUSBFrameProvider.PipelineCandidate;
import org.wpilib.util.PixelFormat;
import org.wpilib.vision.camera.VideoMode;

public class GStreamerPipelineCandidatesTest {
    private static final long ALL_CAPABILITIES =
            GStreamerCaptureJNI.CAP_OPTIONS
                    | GStreamerCaptureJNI.CAP_JPEG_DECODE
                    | GStreamerCaptureJNI.CAP_STATS
                    | GStreamerCaptureJNI.CAP_LUMA_FORMATS;

    private static List<PipelineCandidate> candidates(
            GStreamerBackend backend,
            PixelFormat format,
            int outputMode,
            String captureMode,
            String hardwareDecoder,
            long capabilities) {
        return GStreamerUSBFrameProvider.buildCandidates(
                backend,
                2,
                new VideoMode(format, 1600, 1304, 60),
                "/dev/video0",
                "true",
                outputMode,
                captureMode,
                hardwareDecoder,
                capabilities,
                element -> element.equals("mppjpegdec"));
    }

    @Test
    public void mjpegMonochromeUsesDirectCaptureFirst() {
        var list =
                candidates(
                        GStreamerBackend.QCOM,
                        PixelFormat.kMJPEG,
                        GStreamerCaptureJNI.OUTPUT_GRAY,
                        "auto",
                        "",
                        ALL_CAPABILITIES);
        assertEquals(2, list.size());
        assertEquals("direct", list.get(0).description());
        assertEquals(GStreamerCaptureJNI.OUTPUT_GRAY, list.get(0).outputMode());
        String direct = list.get(0).pipeline();
        assertTrue(direct.contains("image/jpeg,width=1600,height=1304,framerate=60/1"));
        // No element can hold frames between the camera and the appsink.
        assertFalse(direct.contains("queue"));
        assertFalse(direct.contains("jpegparse"));
        assertFalse(direct.contains("jpegdec"));
        assertFalse(direct.contains("videoconvert"));
        assertTrue(direct.contains("max-buffers=1 drop=true"));
        assertEquals("legacy", list.get(1).description());
    }

    @Test
    public void oldNativeLibraryUsesLegacyPipelineOnly() {
        var list =
                candidates(
                        GStreamerBackend.QCOM,
                        PixelFormat.kMJPEG,
                        GStreamerCaptureJNI.OUTPUT_GRAY,
                        "auto",
                        "",
                        0);
        assertEquals(1, list.size());
        assertEquals("legacy", list.get(0).description());
        assertTrue(list.get(0).pipeline().contains("jpegdec"));
    }

    @Test
    public void rawLumaFormatsUseDirectCaptureForMonochromeOutput() {
        for (PixelFormat format :
                new PixelFormat[] {PixelFormat.kYUYV, PixelFormat.kUYVY, PixelFormat.kGray}) {
            var list =
                    candidates(
                            GStreamerBackend.GENERIC,
                            format,
                            GStreamerCaptureJNI.OUTPUT_GRAY,
                            "auto",
                            "",
                            ALL_CAPABILITIES);
            assertEquals("direct", list.get(0).description(), "format " + format);
            assertTrue(list.get(0).pipeline().contains("video/x-raw,format="));
            assertFalse(list.get(0).pipeline().contains("videoconvert"));
        }
    }

    @Test
    public void rawColorOutputKeepsTheConverterPipeline() {
        var list =
                candidates(
                        GStreamerBackend.GENERIC,
                        PixelFormat.kYUYV,
                        GStreamerCaptureJNI.OUTPUT_BGR,
                        "auto",
                        "",
                        ALL_CAPABILITIES);
        assertEquals(1, list.size());
        assertEquals("legacy", list.get(0).description());
        assertTrue(list.get(0).pipeline().contains("videoconvert"));
    }

    @Test
    public void hardwareDecoderIsFirstWhenRequestedAndInstalled() {
        var list =
                candidates(
                        GStreamerBackend.GENERIC,
                        PixelFormat.kMJPEG,
                        GStreamerCaptureJNI.OUTPUT_GRAY,
                        "auto",
                        "mppjpegdec",
                        ALL_CAPABILITIES);
        assertEquals(3, list.size());
        assertTrue(list.get(0).description().contains("mppjpegdec"));
        // Monochrome output uses the luma plane of the decoder output. No converter.
        assertFalse(list.get(0).pipeline().contains("videoconvert"));
        assertTrue(list.get(0).pipeline().contains("leaky=downstream ! mppjpegdec ! appsink"));
        assertEquals("direct", list.get(1).description());
        assertEquals("legacy", list.get(2).description());

        // A decoder that is not installed is ignored.
        var missing =
                candidates(
                        GStreamerBackend.GENERIC,
                        PixelFormat.kMJPEG,
                        GStreamerCaptureJNI.OUTPUT_GRAY,
                        "auto",
                        "v4l2jpegdec",
                        ALL_CAPABILITIES);
        assertEquals("direct", missing.get(0).description());
    }

    @Test
    public void captureModeSettingSelectsOneType() {
        var legacy =
                candidates(
                        GStreamerBackend.QCOM,
                        PixelFormat.kMJPEG,
                        GStreamerCaptureJNI.OUTPUT_BGR,
                        "legacy",
                        "",
                        ALL_CAPABILITIES);
        assertEquals(1, legacy.size());
        assertEquals("legacy", legacy.get(0).description());

        var direct =
                candidates(
                        GStreamerBackend.QCOM,
                        PixelFormat.kMJPEG,
                        GStreamerCaptureJNI.OUTPUT_BGR,
                        "direct",
                        "",
                        ALL_CAPABILITIES);
        assertEquals(1, direct.size());
        assertEquals("direct", direct.get(0).description());
        assertEquals(GStreamerCaptureJNI.OUTPUT_BGR, direct.get(0).outputMode());
    }

    @Test
    public void jetsonKeepsItsPipeline() {
        var list =
                candidates(
                        GStreamerBackend.JETSON,
                        PixelFormat.kMJPEG,
                        GStreamerCaptureJNI.OUTPUT_GRAY,
                        "auto",
                        "",
                        ALL_CAPABILITIES);
        assertEquals(1, list.size());
        assertEquals("jetson", list.get(0).description());
        assertEquals(GStreamerCaptureJNI.OUTPUT_AUTO, list.get(0).outputMode());
    }
}
