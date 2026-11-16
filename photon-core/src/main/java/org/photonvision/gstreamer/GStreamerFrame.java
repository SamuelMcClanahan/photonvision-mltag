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

    public FrameCaptureMetadata toCaptureMetadata(String source, long fallbackTimestampNanos) {
        long captureTimestampNanos =
                estimatedAgeNanos >= 0
                        ? fallbackTimestampNanos - estimatedAgeNanos
                        : fallbackTimestampNanos;

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
}
