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
import org.opencv.imgproc.Imgproc;
import org.photonvision.common.hardware.Platform;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.gstreamer.GStreamerCapture;
import org.photonvision.gstreamer.GStreamerCaptureJNI;
import org.photonvision.gstreamer.GStreamerFrame;
import org.photonvision.vision.frame.FrameCaptureMetadata;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.MatPool;
import org.photonvision.vision.processes.VisionSourceSettables;
import org.wpilib.util.PixelFormat;

/** Native GStreamer USB provider that bypasses CScore camera ownership. */
public class GStreamerUSBFrameProvider extends CpuImageProcessor {
    enum GStreamerBackend {
        JETSON,
        QCOM,
        GENERIC
    }

    /**
     * One way to open the camera. The provider tries the candidates in order and uses the first one
     * that gives a frame.
     */
    record PipelineCandidate(String pipeline, int outputMode, String description) {}

    private static final Object CAPTURE_OPEN_LOCK = new Object();
    private static long lastOpenCompleteMillis = 0;

    /**
     * A capture that gives no frame for this time is closed and opened again. A short USB disconnect
     * ends the GStreamer stream, and without a reopen the camera stays lost until a restart. 0 = off.
     */
    private static final long STALL_REOPEN_NANOS =
            EnvUtils.nonNegativeInteger("PV_USB_GSTREAMER_STALL_REOPEN_MS", 2000) * 1_000_000L;

    /** A grab that returns faster than this did not wait for a frame (for example after EOS). */
    private static final long FAST_EMPTY_GRAB_NANOS = 20_000_000L;

    private final Logger logger;
    private final VisionSourceSettables settables;
    private final Runnable connectedCallback;
    private GStreamerCapture capture = null;
    private boolean captureFailed = false;
    private long lastOpenFailureMillis = 0;
    private long lastFrameNanos = 0;
    private boolean libraryErrorLogged = false;
    private final MatPool colorMatPool = new MatPool();
    private volatile boolean monochromeAllowed = false;

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

        long grabStart = System.nanoTime();
        GStreamerFrame frame = capture.grab(250);
        if (frame == null) {
            long now = System.nanoTime();
            long stallNanos = stallReopenNanos();
            if (stallNanos > 0 && now - lastFrameNanos >= stallNanos) {
                // The camera stopped. Close the capture so that the normal open path with its
                // cooldown opens the camera again.
                logger.warn(
                        "Native GStreamer USB capture gave no frame for "
                                + (now - lastFrameNanos) / 1_000_000L
                                + " ms; opening it again");
                releaseCapture();
                captureFailed = true;
                lastOpenFailureMillis = System.currentTimeMillis();
            } else if (now - grabStart < FAST_EMPTY_GRAB_NANOS) {
                // The native side did not wait (the stream ended). Wait here so that the vision
                // loop does not use a full CPU and make garbage for each empty frame.
                sleepQuietly(FAST_EMPTY_GRAB_NANOS / 1_000_000L);
            }
            return new CapturedFrame(new CVMat(), settables.getFrameStaticProperties(), 0);
        }
        lastFrameNanos = System.nanoTime();

        FrameCaptureMetadata metadata =
                frame.toCaptureMetadata(FrameCaptureMetadata.SOURCE_GSTREAMER_JNI, MathUtils.wpiNanoTime());
        CVMat image = frame.toCVMat();
        if (frame.channels() == 1 && !monochromeAllowed) {
            image = expandToBgr(image);
        }

        return new CapturedFrame(image, settables.getFrameStaticProperties(), metadata);
    }

    /**
     * Makes a 3-channel copy of a 1-channel camera image. Only the AprilTag pipeline accepts a
     * 1-channel input image. All other pipelines get the BGR layout that they expect.
     */
    private CVMat expandToBgr(CVMat gray) {
        CVMat bgr = colorMatPool.take();
        try {
            Imgproc.cvtColor(gray.getMat(), bgr.getMat(), Imgproc.COLOR_GRAY2BGR);
        } finally {
            gray.release();
        }
        return bgr;
    }

    @Override
    public void requestMonochromeInput(boolean allowed) {
        monochromeAllowed = allowed;
    }

    private boolean ensureCapture() {
        if (captureFailed && !shouldRetryOpen()) {
            return false;
        }
        if (capture != null && capture.isOpen()) {
            return true;
        }

        if (!nativeLibraryAvailable()) {
            // Without the library no candidate can open. Say why one time, because the general
            // open warnings do not give the cause.
            if (!libraryErrorLogged) {
                logger.error(
                        "Native GStreamer capture library is not loaded ("
                                + GStreamerCaptureJNI.libraryDescription()
                                + "): "
                                + java.util.Objects.requireNonNullElse(
                                        GStreamerCaptureJNI.loadError(), "unknown load error"));
                libraryErrorLogged = true;
            }
            captureFailed = true;
            lastOpenFailureMillis = System.currentTimeMillis();
            return false;
        }

        int openAttempts = EnvUtils.integer("PV_USB_GSTREAMER_OPEN_ATTEMPTS", 3);
        for (int ioMode : ioModeCandidates()) {
            for (PipelineCandidate candidate : pipelineCandidates(ioMode)) {
                for (int attempt = 1; attempt <= openAttempts; attempt++) {
                    capture = openNative(candidate);
                    if (capture == null || !capture.isOpen()) {
                        releaseCapture();
                        if (attempt < openAttempts) {
                            logger.warn(
                                    "Failed to open native GStreamer USB capture ("
                                            + candidate.description()
                                            + ") with io-mode="
                                            + ioMode
                                            + "; retrying "
                                            + attempt
                                            + "/"
                                            + openAttempts);
                            sleepQuietly(250L * attempt);
                        }
                        continue;
                    }

                    if (waitForFirstFrame()) {
                        logger.info(
                                "Opened native GStreamer USB capture ("
                                        + candidate.description()
                                        + "): "
                                        + candidate.pipeline());
                        captureFailed = false;
                        lastFrameNanos = System.nanoTime();
                        return true;
                    }

                    // The pipeline opened but gave no frame. The same pipeline gives no frame on a
                    // second attempt either, so go to the next candidate.
                    releaseCapture();
                    break;
                }
                logger.warn(
                        "Native GStreamer USB capture candidate gave no frame ("
                                + candidate.description()
                                + "): "
                                + candidate.pipeline());
            }
        }

        logger.error("Failed to open native GStreamer USB capture pipeline; will retry");
        captureFailed = true;
        lastOpenFailureMillis = System.currentTimeMillis();
        return false;
    }

    /**
     * Creates the native pipeline. The global lock covers only the native create call. The cameras
     * do not open at the same time, but a camera that gives no frame does not stop the open of the
     * other cameras.
     */
    private static GStreamerCapture openNative(PipelineCandidate candidate) {
        synchronized (CAPTURE_OPEN_LOCK) {
            staggerOpenIfRequested();
            GStreamerCapture opened =
                    GStreamerCapture.open(candidate.pipeline(), candidate.outputMode());
            if (opened != null) {
                lastOpenCompleteMillis = System.currentTimeMillis();
            }
            return opened;
        }
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

    /**
     * Makes sure that the open capture gives a frame. A pipeline can reach the PLAYING state and then
     * give no frame, or give a pixel format that the native library cannot use. The first frame is
     * the only reliable proof.
     */
    private boolean waitForFirstFrame() {
        long firstFrameTimeoutMs = EnvUtils.integer("PV_USB_GSTREAMER_FIRST_FRAME_TIMEOUT_MS", 2500);
        long deadline = System.currentTimeMillis() + firstFrameTimeoutMs;
        while (System.currentTimeMillis() < deadline) {
            long grabStart = System.nanoTime();
            GStreamerFrame frame = capture.grab(250);
            if (frame != null) {
                // The pipeline is good. This frame is only a test, so release it.
                CVMat mat = frame.toCVMat();
                mat.release();
                return true;
            }
            if (System.nanoTime() - grabStart < FAST_EMPTY_GRAB_NANOS) {
                // The stream ended at once. Do not use a full CPU until the deadline.
                sleepQuietly(FAST_EMPTY_GRAB_NANOS / 1_000_000L);
            }
        }
        return false;
    }

    /** The io-modes to try. Tests override this. */
    int[] ioModeCandidates() {
        return getIoModeCandidates();
    }

    /** The pipelines to try for one io-mode, in order. Tests override this. */
    List<PipelineCandidate> pipelineCandidates(int ioMode) {
        return buildCandidates(ioMode);
    }

    /** True if the native capture library is loaded. Tests override this. */
    boolean nativeLibraryAvailable() {
        return GStreamerCapture.isAvailable();
    }

    /** The time with no frame after which the capture is opened again. Tests override this. */
    long stallReopenNanos() {
        return STALL_REOPEN_NANOS;
    }

    /** The wait after a failed open before the next open. Tests override this. */
    long reopenCooldownMs() {
        return EnvUtils.integer("PV_USB_GSTREAMER_REOPEN_COOLDOWN_MS", 5000);
    }

    private List<PipelineCandidate> buildCandidates(int ioMode) {
        String explicitPipeline = EnvUtils.string("PV_USB_GSTREAMER_PIPELINE", "");
        if (!explicitPipeline.isBlank()) {
            return List.of(
                    new PipelineCandidate(explicitPipeline, requestedOutputMode(), "explicit pipeline"));
        }

        var cameraMode = settables.getCurrentVideoMode();
        String device =
                EnvUtils.string("PV_USB_GSTREAMER_DEVICE", settables.getConfiguration().getDevicePath());
        String doTimestamp = EnvUtils.flag("PV_USB_GSTREAMER_DO_TIMESTAMP", true) ? "true" : "false";
        GStreamerBackend backend = selectBackend();

        return buildCandidates(
                backend,
                ioMode,
                cameraMode,
                device,
                doTimestamp,
                requestedOutputMode(),
                EnvUtils.string("PV_USB_GSTREAMER_CAPTURE_MODE", "auto"),
                EnvUtils.string("PV_USB_GSTREAMER_HW_JPEG_DECODER", ""),
                GStreamerCaptureJNI.capabilities(),
                GStreamerUSBFrameProvider::nativeHasElement);
    }

    private static boolean nativeHasElement(String element) {
        try {
            return GStreamerCaptureJNI.isAvailable() && GStreamerCaptureJNI.hasElement(element);
        } catch (UnsatisfiedLinkError e) {
            return false;
        }
    }

    /** Returns the image layout that the user asked for with PV_USB_GSTREAMER_CPU_FORMAT. */
    private static int requestedOutputMode() {
        String cpuFormat =
                EnvUtils.string("PV_USB_GSTREAMER_CPU_FORMAT", "BGR").trim().toUpperCase(Locale.ROOT);
        return "GRAY8".equals(normalizeCpuFormat(cpuFormat))
                ? GStreamerCaptureJNI.OUTPUT_GRAY
                : GStreamerCaptureJNI.OUTPUT_BGR;
    }

    /**
     * Builds the ordered list of pipelines for one io-mode.
     *
     * <p>Jetson has one pipeline (NVMM). The Qualcomm and generic backends have three types:
     *
     * <ol>
     *   <li>Hardware JPEG decode. Only if PV_USB_GSTREAMER_HW_JPEG_DECODER names an element.
     *   <li>Direct. The camera buffer goes to the appsink with no element between them. The native
     *       library decodes or converts the newest buffer. No stage can hold more than one frame.
     *   <li>Legacy. Software decode and videoconvert in the GStreamer pipeline.
     * </ol>
     */
    static List<PipelineCandidate> buildCandidates(
            GStreamerBackend backend,
            int ioMode,
            org.wpilib.vision.camera.VideoMode cameraMode,
            String device,
            String doTimestamp,
            int outputMode,
            String captureModeSetting,
            String hardwareDecoder,
            long nativeCapabilities,
            java.util.function.Predicate<String> hasElement) {
        List<PipelineCandidate> candidates = new ArrayList<>();
        if (backend == GStreamerBackend.JETSON) {
            candidates.add(
                    new PipelineCandidate(
                            buildJetsonPipeline(ioMode, cameraMode, device, doTimestamp),
                            GStreamerCaptureJNI.OUTPUT_AUTO,
                            "jetson"));
            return candidates;
        }

        boolean gray = outputMode == GStreamerCaptureJNI.OUTPUT_GRAY;
        String captureMode = captureModeSetting.trim().toLowerCase(Locale.ROOT);
        boolean allowDirect = !captureMode.equals("legacy");
        boolean allowLegacy = !captureMode.equals("direct");
        boolean lumaFormats = (nativeCapabilities & GStreamerCaptureJNI.CAP_LUMA_FORMATS) != 0;
        boolean jpegDecode = (nativeCapabilities & GStreamerCaptureJNI.CAP_JPEG_DECODE) != 0;
        boolean compressed = cameraMode.pixelFormat == PixelFormat.kMJPEG;
        String sourceCaps = buildSourceCaps(cameraMode);
        String source =
                String.format(
                        Locale.ROOT,
                        "v4l2src device=%s io-mode=%d do-timestamp=%s ! %s",
                        device,
                        ioMode,
                        doTimestamp,
                        sourceCaps);

        if (allowDirect && compressed) {
            String decoderElement = hardwareDecoder.trim().split("[ !]")[0];
            if (!hardwareDecoder.isBlank() && lumaFormats && hasElement.test(decoderElement)) {
                // The decoder gives a YUV image. In GRAY mode the native library uses the luma plane
                // of that image directly. No converter element is necessary.
                String tail = gray ? "" : " ! videoconvert ! video/x-raw,format=BGR";
                candidates.add(
                        new PipelineCandidate(
                                source
                                        + " ! queue max-size-buffers=1 max-size-time=0 max-size-bytes=0"
                                        + " leaky=downstream ! "
                                        + hardwareDecoder.trim()
                                        + tail
                                        + " ! "
                                        + APPSINK,
                                outputMode,
                                "hardware JPEG decode with " + decoderElement));
            }
        }

        if (allowDirect) {
            boolean direct =
                    switch (cameraMode.pixelFormat) {
                        case kMJPEG -> jpegDecode;
                        // The luma bytes of a raw YUV image are a correct 1-channel image. A BGR image
                        // needs a color conversion, and the legacy pipeline does that.
                        case kYUYV, kUYVY -> lumaFormats && gray;
                        case kGray -> lumaFormats;
                        default -> false;
                    };
            if (direct) {
                candidates.add(new PipelineCandidate(source + " ! " + APPSINK, outputMode, "direct"));
            }
        }

        if (allowLegacy || candidates.isEmpty()) {
            String legacy =
                    backend == GStreamerBackend.QCOM
                            ? buildQcomPipeline(ioMode, cameraMode, device, doTimestamp)
                            : buildGenericPipeline(ioMode, cameraMode, device, doTimestamp);
            candidates.add(new PipelineCandidate(legacy, outputMode, "legacy"));
        }
        return candidates;
    }

    private static final String APPSINK =
            "appsink name=pvappsink emit-signals=false max-buffers=1 drop=true sync=false"
                    + " enable-last-sample=false";

    private static String buildSourceCaps(org.wpilib.vision.camera.VideoMode cameraMode) {
        String rawFormat =
                switch (cameraMode.pixelFormat) {
                    case kYUYV -> "YUY2";
                    case kUYVY -> "UYVY";
                    case kGray -> "GRAY8";
                    default -> null;
                };
        if (rawFormat == null) {
            return String.format(
                    Locale.ROOT,
                    "image/jpeg,width=%d,height=%d,framerate=%d/1",
                    cameraMode.width,
                    cameraMode.height,
                    cameraMode.fps);
        }
        return String.format(
                Locale.ROOT,
                "video/x-raw,format=%s,width=%d,height=%d,framerate=%d/1",
                rawFormat,
                cameraMode.width,
                cameraMode.height,
                cameraMode.fps);
    }

    private static GStreamerBackend selectBackend() {
        String backend = EnvUtils.string("PV_USB_GSTREAMER_BACKEND", "");
        if (!backend.isBlank()) {
            return switch (backend.toLowerCase(Locale.ROOT)) {
                case "jetson", "nvidia", "nvmm" -> GStreamerBackend.JETSON;
                case "qcom", "qualcomm", "rubik", "qcs6490" -> GStreamerBackend.QCOM;
                case "generic", "cpu", "portable" -> GStreamerBackend.GENERIC;
                default ->
                        Platform.isQCS6490()
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
            int ioMode,
            org.wpilib.vision.camera.VideoMode cameraMode,
            String device,
            String doTimestamp) {
        String decoder = EnvUtils.string("PV_USB_GSTREAMER_DECODER", "nvv4l2decoder");
        boolean nvmmTee = EnvUtils.flag("PV_USB_GSTREAMER_NVMM_TEE", true);
        String cpuFormat =
                EnvUtils.string("PV_USB_GSTREAMER_CPU_FORMAT", "BGR").trim().toUpperCase(Locale.ROOT);
        boolean rawYuyv = cameraMode.pixelFormat == PixelFormat.kYUYV;
        String decoderStage = buildNvmmDecoderStage(decoder);
        String jpegParserStage = buildJpegParserStage(decoder);
        String cpuBranchConversion =
                buildCpuBranchConversion(cpuFormat, cameraMode.width, cameraMode.height, cameraMode.fps);

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
            int ioMode,
            org.wpilib.vision.camera.VideoMode cameraMode,
            String device,
            String doTimestamp) {
        String cpuFormat =
                EnvUtils.string("PV_USB_GSTREAMER_CPU_FORMAT", "BGR").trim().toUpperCase(Locale.ROOT);
        boolean rawYuyv = cameraMode.pixelFormat != PixelFormat.kMJPEG;
        String sourceCaps = buildSourceCaps(cameraMode);
        String decodeStage = rawYuyv ? "" : buildQcomDecoderStage();
        String cpuBranchConversion =
                buildQcomCpuBranchConversion(
                        cpuFormat, cameraMode.width, cameraMode.height, cameraMode.fps);

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
            int ioMode,
            org.wpilib.vision.camera.VideoMode cameraMode,
            String device,
            String doTimestamp) {
        String cpuFormat =
                EnvUtils.string("PV_USB_GSTREAMER_CPU_FORMAT", "BGR").trim().toUpperCase(Locale.ROOT);
        boolean rawYuyv = cameraMode.pixelFormat != PixelFormat.kMJPEG;
        String sourceCaps = buildSourceCaps(cameraMode);
        String decodeStage =
                rawYuyv
                        ? ""
                        : stageWithBang(
                                EnvUtils.string("PV_USB_GSTREAMER_DECODE_STAGE", "jpegparse ! jpegdec"));
        String cpuBranchConversion =
                buildGenericCpuBranchConversion(
                        cpuFormat, cameraMode.width, cameraMode.height, cameraMode.fps);

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
                EnvUtils.string(
                        "PV_USB_GSTREAMER_QCOM_DECODER",
                        EnvUtils.string("PV_USB_GSTREAMER_DECODER", "jpegdec"));
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

    private static String buildQcomCpuBranchConversion(
            String cpuFormat, int width, int height, int fps) {
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

    private static String buildGenericCpuBranchConversion(
            String cpuFormat, int width, int height, int fps) {
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
        colorMatPool.release();
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
        return System.currentTimeMillis() - lastOpenFailureMillis >= reopenCooldownMs();
    }
}
