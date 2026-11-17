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

import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;

/** JNI entrypoints for low-latency native GStreamer capture. */
public final class GStreamerCaptureJNI {
    public static final String JNI_LIB_ENV = "PV_GSTREAMER_CAPTURE_JNI_LIB";

    /** Compressed input: decode to BGR. A JPEG with one component gives a 1-channel image. */
    public static final int OUTPUT_AUTO = 0;

    /** Compressed input: decode the luma component only. The result is a 1-channel image. */
    public static final int OUTPUT_GRAY = 1;

    /** Compressed input: decode to BGR. */
    public static final int OUTPUT_BGR = 2;

    /** {@link #createWithOptions(String, int)} is available. */
    public static final long CAP_OPTIONS = 1;

    /** The library decodes {@code image/jpeg} appsink buffers itself. */
    public static final long CAP_JPEG_DECODE = 2;

    /** {@link #getStats(long)} is available. */
    public static final long CAP_STATS = 4;

    /** Planar YUV, packed YUV 4:2:2 and GRAY8 appsink buffers give a 1-channel image. */
    public static final long CAP_LUMA_FORMATS = 8;

    /** {@link #setCurrentThreadCpuMask(long)} is available. */
    public static final long CAP_THREAD_AFFINITY = 16;

    /**
     * {@link #setJpegIdctAccurate(long, boolean)} is available. The JPEG decode can use the accurate
     * or the fast inverse DCT.
     */
    public static final long CAP_JPEG_IDCT = 32;

    private static final Logger logger = new Logger(GStreamerCaptureJNI.class, LogGroup.Camera);

    // Set by load(). It must come before the fields that call load().
    private static String loadError = null;

    private static final boolean loaded = load();
    private static final long capabilities = queryCapabilities();

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
            // Keep the cause. Without it the camera only says that the open failed.
            loadError = String.valueOf(e.getMessage());
            logger.error(
                    "Could not load the native GStreamer capture library ("
                            + libraryDescription()
                            + "): "
                            + loadError);
            return false;
        }
    }

    /** Returns the path or name of the library that load() uses. */
    public static String libraryDescription() {
        String explicitPath = System.getenv(JNI_LIB_ENV);
        if (explicitPath != null && !explicitPath.isBlank()) {
            return JNI_LIB_ENV + "=" + explicitPath;
        }
        return "pv_gstreamer_capture_jni on java.library.path; set " + JNI_LIB_ENV;
    }

    /** Returns the message of the load error, or null if the library is loaded. */
    public static String loadError() {
        return loadError;
    }

    private static long queryCapabilities() {
        if (!loaded) {
            return 0;
        }
        try {
            return getCapabilities();
        } catch (UnsatisfiedLinkError e) {
            // A library that was built before the capability call existed.
            return 0;
        }
    }

    public static boolean isAvailable() {
        return loaded;
    }

    /** Returns the {@code CAP_*} bits of the loaded library. Returns 0 for an old library. */
    public static long capabilities() {
        return capabilities;
    }

    public static boolean hasCapability(long capability) {
        return (capabilities & capability) == capability;
    }

    public static native long create(String pipeline);

    /**
     * Creates a capture.
     *
     * @param pipeline The GStreamer launch string. It must contain {@code appsink name=pvappsink}.
     * @param outputMode One of {@link #OUTPUT_AUTO}, {@link #OUTPUT_GRAY}, {@link #OUTPUT_BGR}.
     */
    public static native long createWithOptions(String pipeline, int outputMode);

    public static native long getCapabilities();

    /**
     * Limits the calling thread to the CPUs of a bit mask (bit n = CPU n). Threads that the calling
     * thread starts after this call get the same limit.
     *
     * @return True if the kernel accepted the mask.
     */
    public static native boolean setCurrentThreadCpuMask(long cpuMask);

    /**
     * Test aid for simulations of a slower processor. After each JPEG frame, the library decodes this
     * percent of the frame again and discards the pixels. 0 = no extra work (default).
     */
    public static native void setTestExtraDecodePercent(int percent);

    /** Returns true if the GStreamer registry of this machine has the element. */
    public static native boolean hasElement(String elementName);

    /**
     * Returns the flow counters of a capture. All values are totals since the capture started.
     *
     * <ol>
     *   <li>buffers that left the source element (camera frames)
     *   <li>samples that left the appsink
     *   <li>frames that were converted or decoded correctly
     *   <li>frames that were given to Java
     *   <li>decoded frames that a newer decoded frame replaced before Java took them
     *   <li>frames that could not be decoded
     *   <li>frames that were rejected because the JPEG data was damaged
     *   <li>sum of the conversion or decode time, nanoseconds
     *   <li>maximum conversion or decode time, nanoseconds
     *   <li>number of frames that have an age value
     *   <li>sum of the frame age at delivery, nanoseconds
     *   <li>maximum frame age at delivery, nanoseconds
     *   <li>sum of the time between the start of a frame and its arrival, nanoseconds
     *   <li>source buffers that had no usable timestamp
     *   <li>delivered frames that had no capture time
     *   <li>buffers that had a timestamp ahead of the pipeline clock
     * </ol>
     */
    public static native long[] getStats(long capturePtr);

    /**
     * Selects the inverse DCT of the JPEG decode for one capture. True: the accurate integer method
     * (the default, nearer to the camera picture). False: the fast method of the GStreamer jpegdec
     * element, which gives the pixels of the old capture pipeline. Only available with {@link
     * #CAP_JPEG_IDCT}.
     */
    public static native void setJpegIdctAccurate(long capturePtr, boolean accurate);

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
     *   <li>frame age in nanoseconds, or -1 if unknown. The age starts at the capture time that a
     *       probe on the source element recorded on the monotonic clock.
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
