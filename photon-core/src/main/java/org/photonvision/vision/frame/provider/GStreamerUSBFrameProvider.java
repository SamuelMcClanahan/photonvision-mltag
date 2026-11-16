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

import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import org.photonvision.common.hardware.Platform;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.gstreamer.GStreamerCapture;
import org.photonvision.gstreamer.GStreamerFrame;
import org.photonvision.vision.frame.FrameCaptureMetadata;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.processes.VisionSourceSettables;
import org.wpilib.util.PixelFormat;

/** Native GStreamer USB provider that bypasses CScore camera ownership. */
public class GStreamerUSBFrameProvider extends CpuImageProcessor {
    private enum GStreamerBackend {
        JETSON,
        QCOM,
        GENERIC
    }

    private static final Object CAPTURE_OPEN_LOCK = new Object();
    private static long lastOpenCompleteMillis = 0;

    private final Logger logger;
    private final VisionSourceSettables settables;
    private final Runnable connectedCallback;
    private GStreamerCapture capture = null;
    private boolean captureFailed = false;
    private boolean pipelineLogged = false;
    private long lastOpenFailureMillis = 0;

    public GStreamerUSBFrameProvider(VisionSourceSettables settables, Runnable connectedCallback) {
        this.settables = settables;
        this.connectedCallback = connectedCallback;
        this.logger =
                new Logger(
                        GStreamerUSBFrameProvider.class,
                        settables.getConfiguration().nickname,
                        LogGroup.Camera);
    }

    @Override
    CapturedFrame getInputMat() {
        return grabGStreamerFrame();
    }

    private CapturedFrame grabGStreamerFrame() {
        if (!ensureCapture()) {
            sleepQuietly(100);
            return new CapturedFrame(new CVMat(), settables.getFrameStaticProperties(), 0);
        }

        GStreamerFrame frame = capture.grab(250);
        if (frame == null) {
            return new CapturedFrame(new CVMat(), settables.getFrameStaticProperties(), 0);
        }

        return new CapturedFrame(
                frame.toCVMat(),
                settables.getFrameStaticProperties(),
                frame.toCaptureMetadata(
                        FrameCaptureMetadata.SOURCE_GSTREAMER_JNI, MathUtils.wpiNanoTime()));
    }

    private boolean ensureCapture() {
        if (captureFailed && !shouldRetryOpen()) {
            return false;
        }
        if (capture != null && capture.isOpen()) {
            return true;
        }

        synchronized (CAPTURE_OPEN_LOCK) {
            staggerOpenIfRequested();

            int openAttempts = EnvUtils.integer("PV_USB_GSTREAMER_OPEN_ATTEMPTS", 3);
            for (int ioMode : getIoModeCandidates()) {
                String pipeline = buildPipeline(ioMode);
                if (!pipelineLogged) {
                    logger.info("Opening native GStreamer USB capture pipeline: " + pipeline);
                    pipelineLogged = true;
                } else {
                    logger.debug("Trying native GStreamer USB capture pipeline: " + pipeline);
                }

                for (int attempt = 1; attempt <= openAttempts; attempt++) {
                    capture = GStreamerCapture.open(pipeline);
                    if (capture != null && capture.isOpen()) {
                        captureFailed = false;
                        lastOpenCompleteMillis = System.currentTimeMillis();
                        return true;
                    }

                    releaseCapture();
                    if (attempt < openAttempts) {
                        logger.warn(
                                "Failed to open native GStreamer USB capture pipeline with io-mode="
                                        + ioMode
                                        + "; retrying "
                                        + attempt
                                        + "/"
                                        + openAttempts);
                        sleepQuietly(250L * attempt);
                    }
                }
            }
        }

        logger.error("Failed to open native GStreamer USB capture pipeline; will retry");
        captureFailed = true;
        lastOpenFailureMillis = System.currentTimeMillis();
        return false;
    }

    private static void staggerOpenIfRequested() {
        long staggerMs = EnvUtils.integer("PV_USB_GSTREAMER_OPEN_STAGGER_MS", 0);
        if (staggerMs <= 0 || lastOpenCompleteMillis <= 0) {
            return;
        }

        long elapsedMs = System.currentTimeMillis() - lastOpenCompleteMillis;
        if (elapsedMs < staggerMs) {
            sleepQuietly(staggerMs - elapsedMs);
        }
    }

    private String buildPipeline(int ioMode) {
        String explicitPipeline = EnvUtils.string("PV_USB_GSTREAMER_PIPELINE", "");
        if (!explicitPipeline.isBlank()) {
            return explicitPipeline;
        }

        var cameraMode = settables.getCurrentVideoMode();
        String device =
                EnvUtils.string("PV_USB_GSTREAMER_DEVICE", settables.getConfiguration().getDevicePath());
        String doTimestamp = EnvUtils.flag("PV_USB_GSTREAMER_DO_TIMESTAMP", true) ? "true" : "false";
        GStreamerBackend backend = selectBackend();

        return switch (backend) {
            case JETSON -> buildJetsonPipeline(ioMode, cameraMode, device, doTimestamp);
            case QCOM -> buildQcomPipeline(ioMode, cameraMode, device, doTimestamp);
            case GENERIC -> buildGenericPipeline(ioMode, cameraMode, device, doTimestamp);
        };
    }

    private static GStreamerBackend selectBackend() {
        String backend = EnvUtils.string("PV_USB_GSTREAMER_BACKEND", "");
        if (!backend.isBlank()) {
            return switch (backend.toLowerCase(Locale.ROOT)) {
                case "jetson", "nvidia", "nvmm" -> GStreamerBackend.JETSON;
                case "qcom", "qualcomm", "rubik", "qcs6490" -> GStreamerBackend.QCOM;
                case "generic", "cpu", "portable" -> GStreamerBackend.GENERIC;
                default -> Platform.isQCS6490()
                        ? GStreamerBackend.QCOM
                        : Platform.isJetson() ? GStreamerBackend.JETSON : GStreamerBackend.GENERIC;
            };
        }

        if (Platform.isQCS6490()) {
            return GStreamerBackend.QCOM;
        }
        if (Platform.isJetson()) {
            return GStreamerBackend.JETSON;
        }
        return GStreamerBackend.GENERIC;
    }

    private static String buildJetsonPipeline(
            int ioMode, org.wpilib.vision.camera.VideoMode cameraMode, String device, String doTimestamp) {
        String decoder = EnvUtils.string("PV_USB_GSTREAMER_DECODER", "nvv4l2decoder");
        boolean nvmmTee = EnvUtils.flag("PV_USB_GSTREAMER_NVMM_TEE", true);
        String cpuFormat =
                EnvUtils.string("PV_USB_GSTREAMER_CPU_FORMAT", "BGR").trim().toUpperCase(Locale.ROOT);
        boolean rawYuyv = cameraMode.pixelFormat == PixelFormat.kYUYV;
        String decoderStage = buildNvmmDecoderStage(decoder);
        String jpegParserStage = buildJpegParserStage(decoder);
        String cpuBranchConversion = buildCpuBranchConversion(cpuFormat, cameraMode.width, cameraMode.height, cameraMode.fps);

        if (!nvmmTee) {
            if (rawYuyv) {
                return String.format(
                        Locale.ROOT,
                        "v4l2src device=%s io-mode=%d do-timestamp=%s ! video/x-raw,format=YUY2,width=%d,height=%d,framerate=%d/1 ! queue max-size-buffers=1 leaky=downstream ! %s ! appsink name=pvappsink emit-signals=false max-buffers=1 drop=true sync=false",
                        device,
                        ioMode,
                        doTimestamp,
                        cameraMode.width,
                        cameraMode.height,
                        cameraMode.fps,
                        cpuBranchConversion);
            }
            return String.format(
                    Locale.ROOT,
                    "v4l2src device=%s io-mode=%d do-timestamp=%s ! image/jpeg,width=%d,height=%d,framerate=%d/1 ! queue max-size-buffers=1 leaky=downstream ! %s%s ! %s ! appsink name=pvappsink emit-signals=false max-buffers=1 drop=true sync=false",
                    device,
                    ioMode,
                    doTimestamp,
                    cameraMode.width,
                    cameraMode.height,
                    cameraMode.fps,
                    jpegParserStage,
                    decoderStage,
                    cpuBranchConversion);
        }

        String mlBranch =
                String.format(
                        Locale.ROOT,
                        "pvtee. ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! nvvidconv ! video/x-raw(memory:NVMM),format=RGBA,width=%d,height=%d,framerate=%d/1 ! appsink name=pvmlsink emit-signals=false max-buffers=1 drop=true sync=false",
                        cameraMode.width,
                        cameraMode.height,
                        cameraMode.fps);
        String cpuBranch =
                String.format(
                        Locale.ROOT,
                        "pvtee. ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! %s ! appsink name=pvappsink emit-signals=false max-buffers=1 drop=true sync=false",
                        cpuBranchConversion);
        if (rawYuyv) {
            return String.format(
                    Locale.ROOT,
                    "v4l2src device=%s io-mode=%d do-timestamp=%s ! video/x-raw,format=YUY2,width=%d,height=%d,framerate=%d/1 ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! nvvidconv ! video/x-raw(memory:NVMM),format=RGBA,width=%d,height=%d,framerate=%d/1 ! tee name=pvtee %s %s",
                    device,
                    ioMode,
                    doTimestamp,
                    cameraMode.width,
                    cameraMode.height,
                    cameraMode.fps,
                    cameraMode.width,
                    cameraMode.height,
                    cameraMode.fps,
                    mlBranch,
                    cpuBranch);
        }

        return String.format(
                Locale.ROOT,
                "v4l2src device=%s io-mode=%d do-timestamp=%s ! image/jpeg,width=%d,height=%d,framerate=%d/1 ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! %s%s ! tee name=pvtee %s %s",
                device,
                ioMode,
                doTimestamp,
                cameraMode.width,
                cameraMode.height,
                cameraMode.fps,
                jpegParserStage,
                decoderStage,
                mlBranch,
                cpuBranch);
    }

    private static String buildQcomPipeline(
            int ioMode, org.wpilib.vision.camera.VideoMode cameraMode, String device, String doTimestamp) {
        String cpuFormat =
                EnvUtils.string("PV_USB_GSTREAMER_CPU_FORMAT", "BGR").trim().toUpperCase(Locale.ROOT);
        boolean rawYuyv = cameraMode.pixelFormat == PixelFormat.kYUYV;
        String sourceCaps =
                rawYuyv
                        ? String.format(
                                Locale.ROOT,
                                "video/x-raw,format=YUY2,width=%d,height=%d,framerate=%d/1",
                                cameraMode.width,
                                cameraMode.height,
                                cameraMode.fps)
                        : String.format(
                                Locale.ROOT,
                                "image/jpeg,width=%d,height=%d,framerate=%d/1",
                                cameraMode.width,
                                cameraMode.height,
                                cameraMode.fps);
        String decodeStage = rawYuyv ? "" : buildQcomDecoderStage();
        String cpuBranchConversion =
                buildQcomCpuBranchConversion(cpuFormat, cameraMode.width, cameraMode.height, cameraMode.fps);

        if (!EnvUtils.flag("PV_USB_GSTREAMER_QCOM_TEE", false)) {
            return String.format(
                    Locale.ROOT,
                    "v4l2src device=%s io-mode=%d do-timestamp=%s ! %s ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! %s%s ! appsink name=pvappsink emit-signals=false max-buffers=1 drop=true sync=false",
                    device,
                    ioMode,
                    doTimestamp,
                    sourceCaps,
                    decodeStage,
                    cpuBranchConversion);
        }

        String mlBranch =
                String.format(
                        Locale.ROOT,
                        "pvtee. ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! %s ! appsink name=pvmlsink emit-signals=false max-buffers=1 drop=true sync=false",
                        buildQcomMlBranchConversion(cameraMode.width, cameraMode.height, cameraMode.fps));
        String cpuBranch =
                String.format(
                        Locale.ROOT,
                        "pvtee. ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! %s ! appsink name=pvappsink emit-signals=false max-buffers=1 drop=true sync=false",
                        cpuBranchConversion);
        return String.format(
                Locale.ROOT,
                "v4l2src device=%s io-mode=%d do-timestamp=%s ! %s ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! %s tee name=pvtee %s %s",
                device,
                ioMode,
                doTimestamp,
                sourceCaps,
                decodeStage,
                mlBranch,
                cpuBranch);
    }

    private static String buildGenericPipeline(
            int ioMode, org.wpilib.vision.camera.VideoMode cameraMode, String device, String doTimestamp) {
        String cpuFormat =
                EnvUtils.string("PV_USB_GSTREAMER_CPU_FORMAT", "BGR").trim().toUpperCase(Locale.ROOT);
        boolean rawYuyv = cameraMode.pixelFormat == PixelFormat.kYUYV;
        String sourceCaps =
                rawYuyv
                        ? String.format(
                                Locale.ROOT,
                                "video/x-raw,format=YUY2,width=%d,height=%d,framerate=%d/1",
                                cameraMode.width,
                                cameraMode.height,
                                cameraMode.fps)
                        : String.format(
                                Locale.ROOT,
                                "image/jpeg,width=%d,height=%d,framerate=%d/1",
                                cameraMode.width,
                                cameraMode.height,
                                cameraMode.fps);
        String decodeStage =
                rawYuyv
                        ? ""
                        : stageWithBang(
                                EnvUtils.string("PV_USB_GSTREAMER_DECODE_STAGE", "jpegparse ! jpegdec"));
        String cpuBranchConversion =
                buildGenericCpuBranchConversion(cpuFormat, cameraMode.width, cameraMode.height, cameraMode.fps);

        return String.format(
                Locale.ROOT,
                "v4l2src device=%s io-mode=%d do-timestamp=%s ! %s ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream ! %s%s ! appsink name=pvappsink emit-signals=false max-buffers=1 drop=true sync=false",
                device,
                ioMode,
                doTimestamp,
                sourceCaps,
                decodeStage,
                cpuBranchConversion);
    }

    private static int[] getIoModeCandidates() {
        if (EnvUtils.hasValue("PV_USB_GSTREAMER_PIPELINE")) {
            return new int[] {EnvUtils.integer("PV_USB_GSTREAMER_IO_MODE", 4)};
        }

        if (EnvUtils.hasValue("PV_USB_GSTREAMER_IO_MODE")) {
            return new int[] {EnvUtils.integer("PV_USB_GSTREAMER_IO_MODE", 4)};
        }

        String rawModes = System.getenv("PV_USB_GSTREAMER_IO_MODES");
        if (rawModes != null && !rawModes.isBlank()) {
            List<Integer> modes = new ArrayList<>();
            for (String rawMode : rawModes.split(",")) {
                try {
                    modes.add(Integer.parseInt(rawMode.trim()));
                } catch (NumberFormatException ignored) {
                    // Skip malformed candidates and keep any usable modes.
                }
            }
            if (!modes.isEmpty()) {
                return modes.stream().mapToInt(Integer::intValue).toArray();
            }
        }

        // DMABUF mode avoids occasional V4L2 buffer-pool activation failures on Jetson USB UVC.
        return new int[] {4, 2, 0};
    }

    private static String buildNvmmDecoderStage(String decoder) {
        return switch (decoder.toLowerCase(Locale.ROOT)) {
            case "nvjpeg", "nvjpegdec" -> "nvjpegdec";
            default ->
                    String.format(
                            Locale.ROOT,
                            "nvv4l2decoder mjpeg=1 disable-dpb=true enable-max-performance=true num-extra-surfaces=%d",
                            EnvUtils.integer("PV_USB_GSTREAMER_NVV4L2_EXTRA_SURFACES", 0));
        };
    }

    private static String buildJpegParserStage(String decoder) {
        if (EnvUtils.hasValue("PV_USB_GSTREAMER_JPEGPARSE")) {
            return EnvUtils.flag("PV_USB_GSTREAMER_JPEGPARSE") ? "jpegparse ! " : "";
        }
        String normalized = decoder.toLowerCase(Locale.ROOT);
        return normalized.equals("nvv4l2decoder") ? "" : "jpegparse ! ";
    }

    private static String buildCpuBranchConversion(String cpuFormat, int width, int height, int fps) {
        if ("GRAY8".equals(cpuFormat) || "GREY8".equals(cpuFormat)) {
            return String.format(
                    Locale.ROOT,
                    "nvvidconv ! video/x-raw,format=GRAY8,width=%d,height=%d,framerate=%d/1",
                    width,
                    height,
                    fps);
        }

        return String.format(
                Locale.ROOT,
                "nvvidconv ! video/x-raw,format=BGRx,width=%d,height=%d,framerate=%d/1 ! videoconvert ! video/x-raw,format=BGR",
                width,
                height,
                fps);
    }

    private static String buildQcomDecoderStage() {
        String explicitStage = EnvUtils.string("PV_USB_GSTREAMER_QCOM_DECODE_STAGE", "");
        if (!explicitStage.isBlank()) {
            return stageWithBang(explicitStage);
        }

        String decoder =
                EnvUtils.string("PV_USB_GSTREAMER_QCOM_DECODER", EnvUtils.string("PV_USB_GSTREAMER_DECODER", "jpegdec"));
        String parserStage = buildQcomJpegParserStage(decoder);
        return parserStage + stageWithBang(decoder);
    }

    private static String buildQcomJpegParserStage(String decoder) {
        if (EnvUtils.hasValue("PV_USB_GSTREAMER_QCOM_JPEGPARSE")) {
            return EnvUtils.flag("PV_USB_GSTREAMER_QCOM_JPEGPARSE") ? "jpegparse ! " : "";
        }
        if (EnvUtils.hasValue("PV_USB_GSTREAMER_JPEGPARSE")) {
            return EnvUtils.flag("PV_USB_GSTREAMER_JPEGPARSE") ? "jpegparse ! " : "";
        }

        String normalized = decoder.toLowerCase(Locale.ROOT);
        return normalized.equals("jpegdec") || normalized.contains("jpeg") ? "jpegparse ! " : "";
    }

    private static String buildQcomCpuBranchConversion(String cpuFormat, int width, int height, int fps) {
        String explicitStage = EnvUtils.string("PV_USB_GSTREAMER_QCOM_CPU_STAGE", "");
        if (!explicitStage.isBlank()) {
            return explicitStage;
        }

        String converter =
                EnvUtils.string(
                        "PV_USB_GSTREAMER_QCOM_CONVERTER",
                        EnvUtils.string("PV_USB_GSTREAMER_CONVERTER", "videoconvert"));
        String targetFormat = normalizeCpuFormat(cpuFormat);
        return String.format(
                Locale.ROOT,
                "%s ! video/x-raw,format=%s,width=%d,height=%d,framerate=%d/1",
                converter,
                targetFormat,
                width,
                height,
                fps);
    }

    private static String buildQcomMlBranchConversion(int width, int height, int fps) {
        String explicitStage = EnvUtils.string("PV_USB_GSTREAMER_QCOM_ML_STAGE", "");
        if (!explicitStage.isBlank()) {
            return explicitStage;
        }

        String converter =
                EnvUtils.string(
                        "PV_USB_GSTREAMER_QCOM_CONVERTER",
                        EnvUtils.string("PV_USB_GSTREAMER_CONVERTER", "videoconvert"));
        String caps =
                EnvUtils.string(
                        "PV_USB_GSTREAMER_QCOM_ML_CAPS",
                        String.format(
                                Locale.ROOT,
                                "video/x-raw,format=RGB,width=%d,height=%d,framerate=%d/1",
                                width,
                                height,
                                fps));
        return converter + " ! " + caps;
    }

    private static String buildGenericCpuBranchConversion(String cpuFormat, int width, int height, int fps) {
        String explicitStage = EnvUtils.string("PV_USB_GSTREAMER_CPU_STAGE", "");
        if (!explicitStage.isBlank()) {
            return explicitStage;
        }

        return String.format(
                Locale.ROOT,
                "videoconvert ! video/x-raw,format=%s,width=%d,height=%d,framerate=%d/1",
                normalizeCpuFormat(cpuFormat),
                width,
                height,
                fps);
    }

    private static String normalizeCpuFormat(String cpuFormat) {
        if ("GRAY8".equals(cpuFormat) || "GREY8".equals(cpuFormat)) {
            return "GRAY8";
        }
        if ("BGRX".equals(cpuFormat) || "BGRx".equals(cpuFormat)) {
            return "BGRx";
        }
        return "BGR";
    }

    private static String stageWithBang(String stage) {
        String trimmed = stage.trim();
        if (trimmed.isEmpty()) {
            return "";
        }
        return trimmed.endsWith("!") ? trimmed + " " : trimmed + " ! ";
    }

    private static void sleepQuietly(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }

    @Override
    public String getName() {
        return "GStreamerUSBFrameProvider - " + settables.getConfiguration().nickname;
    }

    @Override
    public void release() {
        releaseCapture();
        releaseProcessorResources();
    }

    private void releaseCapture() {
        if (capture != null) {
            capture.close();
            capture = null;
        }
    }

    @Override
    protected void onCameraConnected() {
        logger.info("GStreamer camera connected! running callback");
        super.onCameraConnected();
        connectedCallback.run();
    }

    @Override
    protected boolean checkCameraConnected() {
        return !captureFailed || shouldRetryOpen();
    }

    private boolean shouldRetryOpen() {
        long cooldownMs = EnvUtils.integer("PV_USB_GSTREAMER_REOPEN_COOLDOWN_MS", 5000);
        return System.currentTimeMillis() - lastOpenFailureMillis >= cooldownMs;
    }
}
