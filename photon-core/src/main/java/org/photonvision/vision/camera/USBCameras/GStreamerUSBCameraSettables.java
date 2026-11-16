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

package org.photonvision.vision.camera.USBCameras;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.HashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.photonvision.common.configuration.CameraConfiguration;
import org.photonvision.common.util.EnvUtils;
import org.photonvision.vision.processes.VisionSourceSettables;
import org.wpilib.util.PixelFormat;
import org.wpilib.vision.camera.VideoMode;

/** Minimal USB settables for the Jetson GStreamer benchmark path. */
public class GStreamerUSBCameraSettables extends VisionSourceSettables {
    private static final Pattern CONTROL_RANGE_PATTERN =
            Pattern.compile("\\s*([\\w_]+)\\s+.*min=(-?\\d+)\\s+max=(-?\\d+).*");
    private static final int PROP_AUTO_EXPOSURE_ENABLED = 3;
    private static final int PROP_AUTO_EXPOSURE_DISABLED = 1;

    private VideoMode currentVideoMode;
    private final String devicePath;
    private final Map<String, ControlRange> controlRanges = new HashMap<>();
    private double lastExposureRaw = 20;
    private int lastBrightness = 0;
    private int lastGain = 0;
    private int lastWhiteBalanceTemp = 4000;

    public GStreamerUSBCameraSettables(CameraConfiguration configuration) {
        super(configuration);
        devicePath = resolveDevicePath(configuration);

        currentVideoMode =
                new VideoMode(
                        parsePixelFormatEnv(),
                        EnvUtils.integer("PV_USB_GSTREAMER_WIDTH", 1280),
                        EnvUtils.integer("PV_USB_GSTREAMER_HEIGHT", 800),
                        EnvUtils.integer("PV_USB_GSTREAMER_FPS", 120));
        videoModes.put(0, currentVideoMode);
        onCameraConnected();
    }

    @Override
    public void onCameraConnected() {
        super.onCameraConnected();
        cacheControlRanges();
        setControlIfPresent("exposure_dynamic_framerate", 0);
        setControlIfPresent("gain_automatic", 0);
        calculateFrameStaticProps();
    }

    @Override
    public void setExposureRaw(double exposureRaw) {
        if (exposureRaw < 0.0) {
            return;
        }

        int value = clampToControl("exposure_time_absolute", (int) Math.round(exposureRaw));
        logger.debug(
                "Setting GStreamer V4L2 exposure_time_absolute on "
                        + devicePath
                        + " to "
                        + value
                        + " (user requested "
                        + exposureRaw
                        + ")");
        setControlIfPresent("auto_exposure", PROP_AUTO_EXPOSURE_DISABLED);
        setControl("exposure_time_absolute", value);
        lastExposureRaw = exposureRaw;
    }

    @Override
    public double getMinExposureRaw() {
        return getControlRange("exposure_time_absolute").map(ControlRange::min).orElse(1);
    }

    @Override
    public double getMaxExposureRaw() {
        return getControlRange("exposure_time_absolute").map(ControlRange::max).orElse(5000);
    }

    @Override
    public void setAutoExposure(boolean cameraAutoExposure) {
        logger.debug(
                "Setting GStreamer V4L2 auto exposure on " + devicePath + " to " + cameraAutoExposure);
        setControl(
                "auto_exposure",
                cameraAutoExposure ? PROP_AUTO_EXPOSURE_ENABLED : PROP_AUTO_EXPOSURE_DISABLED);

        if (!cameraAutoExposure) {
            setExposureRaw(lastExposureRaw);
        }
    }

    @Override
    public void setWhiteBalanceTemp(double temp) {
        int value = clampToControl("white_balance_temperature", (int) Math.round(temp));
        setControlIfPresent("white_balance_automatic", 0);
        if (setControlIfPresent("white_balance_temperature", value)) {
            lastWhiteBalanceTemp = value;
        }
    }

    @Override
    public void setAutoWhiteBalance(boolean autowb) {
        setControlIfPresent("white_balance_automatic", autowb ? 1 : 0);
        if (!autowb) {
            setWhiteBalanceTemp(lastWhiteBalanceTemp);
        }
    }

    @Override
    public void setBrightness(int brightness) {
        int value = clampToControl("brightness", brightness);
        if (setControlIfPresent("brightness", value)) {
            lastBrightness = value;
        }
    }

    @Override
    public void setGain(int gain) {
        int value = clampToControl("gain", gain);
        setControlIfPresent("gain_automatic", 0);
        if (setControlIfPresent("gain", value)) {
            lastGain = value;
        }
    }

    @Override
    public VideoMode getCurrentVideoMode() {
        return currentVideoMode;
    }

    @Override
    protected void setVideoModeInternal(VideoMode videoMode) {
        if (videoMode != null) {
            currentVideoMode = videoMode;
            videoModes.put(0, videoMode);
        }
    }

    @Override
    public HashMap<Integer, VideoMode> getAllVideoModes() {
        return videoModes;
    }

    @Override
    public double getMinWhiteBalanceTemp() {
        return 1;
    }

    @Override
    public double getMaxWhiteBalanceTemp() {
        return 4000;
    }

    private static PixelFormat parsePixelFormatEnv() {
        String normalized =
                EnvUtils.string("PV_USB_GSTREAMER_PIXEL_FORMAT", "MJPEG").toUpperCase(Locale.ROOT);
        return switch (normalized) {
            case "YUYV", "YUY2" -> PixelFormat.kYUYV;
            case "UYVY" -> PixelFormat.kUYVY;
            case "GRAY", "GREY", "GRAY8", "Y8", "Y800" -> PixelFormat.kGray;
            case "MJPEG", "MJPG" -> PixelFormat.kMJPEG;
            default -> PixelFormat.kMJPEG;
        };
    }

    private static String resolveDevicePath(CameraConfiguration configuration) {
        return EnvUtils.string("PV_USB_GSTREAMER_DEVICE", configuration.getDevicePath());
    }

    private Optional<ControlRange> getControlRange(String control) {
        return Optional.ofNullable(controlRanges.get(control));
    }

    private int clampToControl(String control, int value) {
        ControlRange range = controlRanges.get(control);
        if (range == null) {
            return value;
        }

        return (int) Math.clamp(value, range.min(), range.max());
    }

    private void cacheControlRanges() {
        controlRanges.clear();
        ProcessBuilder builder = new ProcessBuilder("v4l2-ctl", "-d", devicePath, "--list-ctrls");
        try {
            Process process = builder.start();
            try (BufferedReader reader =
                    new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    Matcher matcher = CONTROL_RANGE_PATTERN.matcher(line);
                    if (!matcher.matches()) {
                        continue;
                    }

                    controlRanges.put(
                            matcher.group(1),
                            new ControlRange(
                                    Integer.parseInt(matcher.group(2)), Integer.parseInt(matcher.group(3))));
                }
            }
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                logger.warn(
                        "Failed to query GStreamer V4L2 controls for "
                                + devicePath
                                + ": v4l2-ctl exited "
                                + exitCode);
            }
        } catch (IOException e) {
            logger.warn("Unable to run v4l2-ctl for GStreamer camera controls on " + devicePath);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.warn("Interrupted while querying GStreamer V4L2 controls on " + devicePath);
        }
    }

    private boolean setControlIfPresent(String control, int value) {
        if (!controlRanges.isEmpty() && !controlRanges.containsKey(control)) {
            logger.debug("No GStreamer V4L2 control " + control + " for " + devicePath + ", skipping.");
            return false;
        }

        return setControl(control, value);
    }

    private boolean setControl(String control, int value) {
        String assignment = String.format(Locale.ROOT, "%s=%d", control, value);
        ProcessBuilder builder =
                new ProcessBuilder("v4l2-ctl", "-d", devicePath, "--set-ctrl", assignment);
        try {
            Process process = builder.start();
            int exitCode = process.waitFor();
            if (exitCode != 0) {
                logger.warn(
                        "Failed to set GStreamer V4L2 control "
                                + assignment
                                + " on "
                                + devicePath
                                + ": v4l2-ctl exited "
                                + exitCode);
                return false;
            }
            return true;
        } catch (IOException e) {
            logger.warn(
                    "Unable to run v4l2-ctl while setting GStreamer V4L2 control "
                            + assignment
                            + " on "
                            + devicePath);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            logger.warn(
                    "Interrupted while setting GStreamer V4L2 control " + assignment + " on " + devicePath);
        }

        return false;
    }

    private record ControlRange(int min, int max) {}
}
