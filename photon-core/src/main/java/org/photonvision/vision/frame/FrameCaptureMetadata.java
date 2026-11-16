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

package org.photonvision.vision.frame;

/** Capture timing metadata carried from camera/native capture into pipeline results. */
public record FrameCaptureMetadata(
        long captureTimestampNanos,
        String source,
        long estimatedAgeNanos,
        long bufferPtsNanos,
        long bufferDtsNanos,
        long bufferDurationNanos,
        long nativePullTimestampNanos,
        long nativeSequence,
        long mlBufferPtsNanos,
        long mlBufferDtsNanos,
        long mlBufferDurationNanos) {
    public static final String SOURCE_UNKNOWN = "unknown";
    public static final String SOURCE_CSCORE = "cscore";
    public static final String SOURCE_CSCORE_RAW = "cscore-raw";
    public static final String SOURCE_GSTREAMER_OPENCV = "gstreamer-opencv";
    public static final String SOURCE_GSTREAMER_JNI = "gstreamer-jni";
    public static final String SOURCE_FILE = "file";
    public static final String SOURCE_LIBCAMERA_GPU = "libcamera-gpu";

    private static final long UNSET = -1;

    public static FrameCaptureMetadata unknown(long captureTimestampNanos) {
        return simple(SOURCE_UNKNOWN, captureTimestampNanos);
    }

    public static FrameCaptureMetadata simple(String source, long captureTimestampNanos) {
        return new FrameCaptureMetadata(
                captureTimestampNanos,
                source,
                UNSET,
                UNSET,
                UNSET,
                UNSET,
                UNSET,
                UNSET,
                UNSET,
                UNSET,
                UNSET);
    }

    public static FrameCaptureMetadata gstreamer(
            String source,
            long captureTimestampNanos,
            long estimatedAgeNanos,
            long bufferPtsNanos,
            long bufferDtsNanos,
            long bufferDurationNanos,
            long nativePullTimestampNanos,
            long nativeSequence,
            long mlBufferPtsNanos,
            long mlBufferDtsNanos,
            long mlBufferDurationNanos) {
        return new FrameCaptureMetadata(
                captureTimestampNanos,
                source,
                estimatedAgeNanos,
                bufferPtsNanos,
                bufferDtsNanos,
                bufferDurationNanos,
                nativePullTimestampNanos,
                nativeSequence,
                mlBufferPtsNanos,
                mlBufferDtsNanos,
                mlBufferDurationNanos);
    }
}
