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

import org.opencv.core.Mat;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.vision.frame.FrameCaptureMetadata;
import org.photonvision.vision.opencv.CVMat;

/** A native GStreamer appsink frame and its timing metadata. */
public record GStreamerFrame(
        long nativeFrameHandle,
        long matNativeObjAddr,
        int width,
        int height,
        int channels,
        long estimatedAgeNanos,
        long bufferPtsNanos,
        long bufferDtsNanos,
        long bufferDurationNanos,
        long nativePullTimestampNanos,
        long sequence,
        long mlBufferPtsNanos,
        long mlBufferDtsNanos,
        long mlBufferDurationNanos) {
    private static final int MIN_FIELD_COUNT = 11;

    public static GStreamerFrame fromNative(long[] fields) {
        if (fields == null || fields.length < MIN_FIELD_COUNT || fields[0] == 0 || fields[1] == 0) {
            return null;
        }

        return new GStreamerFrame(
                fields[0],
                fields[1],
                Math.toIntExact(fields[2]),
                Math.toIntExact(fields[3]),
                Math.toIntExact(fields[4]),
                fields[5],
                fields[6],
                fields[7],
                fields[8],
                fields[9],
                fields[10],
                fields.length > 11 ? fields[11] : -1,
                fields.length > 12 ? fields[12] : -1,
                fields.length > 13 ? fields[13] : -1);
    }

    public CVMat toCVMat() {
        return new CVMat(
                new Mat(matNativeObjAddr),
                () -> GStreamerCaptureJNI.releaseFrame(nativeFrameHandle),
                true,
                nativeFrameHandle);
    }

    /**
     * Makes the capture metadata. The capture time comes only from native clock values, so a JVM
     * pause after grab() returns does not change it.
     */
    public FrameCaptureMetadata toCaptureMetadata(String source) {
        // The native capture time on the monotonic clock. Without a known age the pull time is the
        // best value that we have.
        long nativeCaptureNanos =
                estimatedAgeNanos >= 0
                        ? nativePullTimestampNanos - estimatedAgeNanos
                        : nativePullTimestampNanos;
        long captureTimestampNanos = nativeCaptureNanos + monotonicToWpiOffsetNanos();

        return FrameCaptureMetadata.gstreamer(
                source,
                captureTimestampNanos,
                estimatedAgeNanos,
                bufferPtsNanos,
                bufferDtsNanos,
                bufferDurationNanos,
                nativePullTimestampNanos,
                sequence,
                mlBufferPtsNanos,
                mlBufferDtsNanos,
                mlBufferDurationNanos);
    }

    /** Two reads of System.nanoTime() farther apart than this show a pause between the reads. */
    private static final long MAX_OFFSET_READ_GAP_NANOS = 1_000_000L;

    private static final int MAX_OFFSET_READS = 5;

    /**
     * The WPILib time minus the monotonic clock. On Linux System.nanoTime() reads CLOCK_MONOTONIC,
     * the clock of the native library. A pause between the reads would move the offset, so the WPILib
     * read is between two monotonic reads and a pair with a large gap is read again.
     */
    static long monotonicToWpiOffsetNanos() {
        long bestOffset = 0;
        long bestGap = Long.MAX_VALUE;
        for (int i = 0; i < MAX_OFFSET_READS; i++) {
            long before = System.nanoTime();
            long wpi = MathUtils.wpiNanoTime();
            long after = System.nanoTime();
            long gap = after - before;
            if (gap < bestGap) {
                bestGap = gap;
                bestOffset = wpi - (before + gap / 2);
            }
            if (gap <= MAX_OFFSET_READ_GAP_NANOS) {
                break;
            }
        }
        return bestOffset;
    }
}
