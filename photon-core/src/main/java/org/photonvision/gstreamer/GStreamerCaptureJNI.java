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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.gstreamer;

/** JNI entrypoints for low-latency Jetson GStreamer capture. */
public final class GStreamerCaptureJNI {
    public static final String JNI_LIB_ENV = "PV_GSTREAMER_CAPTURE_JNI_LIB";
    private static final boolean loaded = load();

    private GStreamerCaptureJNI() {}

    private static boolean load() {
        String explicitPath = System.getenv(JNI_LIB_ENV);
        try {
            if (explicitPath != null && !explicitPath.isBlank()) {
                System.load(explicitPath);
            } else {
                System.loadLibrary("pv_gstreamer_capture_jni");
            }
            return true;
        } catch (UnsatisfiedLinkError | SecurityException e) {
            return false;
        }
    }

    public static boolean isAvailable() {
        return loaded;
    }

    public static native long create(String pipeline);

    public static native void destroy(long capturePtr);

    /**
     * Pulls a frame from the native appsink.
     *
     * <p>Layout:
     *
     * <ol>
     *   <li>native frame handle
     *   <li>native {@code cv::Mat*} address
     *   <li>frame width
     *   <li>frame height
     *   <li>frame channels
     *   <li>estimated age in nanoseconds, computed from GstBuffer PTS and the pipeline clock
     *   <li>GstBuffer PTS in nanoseconds, or -1 if invalid
     *   <li>GstBuffer DTS in nanoseconds, or -1 if invalid
     *   <li>GstBuffer duration in nanoseconds, or -1 if invalid
     *   <li>monotonic pull timestamp in nanoseconds, native steady-clock domain
     *   <li>native sequence counter
     * </ol>
     */
    public static native long[] grab(long capturePtr, long timeoutMillis);

    public static native void releaseFrame(long framePtr);
}
