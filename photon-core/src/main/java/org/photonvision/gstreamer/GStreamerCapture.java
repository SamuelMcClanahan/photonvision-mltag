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

/** Owns a native GStreamer pipeline and appsink. */
public final class GStreamerCapture implements AutoCloseable {
    /** Flow counters of a capture. All values are totals since the capture started. */
    public record Stats(
            long sourceBuffers,
            long samplesPulled,
            long framesProduced,
            long framesDelivered,
            long framesReplaced,
            long decodeErrors,
            long corruptFrames,
            long convertNanosSum,
            long convertNanosMax,
            long ageSamples,
            long ageNanosSum,
            long ageNanosMax,
            long deviceDelayNanosSum,
            long sourcePtsInvalid,
            long captureTimeMissing,
            long ptsAheadOfClock) {
        static Stats fromNative(long[] fields) {
            if (fields == null || fields.length < 16) {
                return null;
            }
            return new Stats(
                    fields[0],
                    fields[1],
                    fields[2],
                    fields[3],
                    fields[4],
                    fields[5],
                    fields[6],
                    fields[7],
                    fields[8],
                    fields[9],
                    fields[10],
                    fields[11],
                    fields[12],
                    fields[13],
                    fields[14],
                    fields[15]);
        }
    }

    private long nativePtr;

    private GStreamerCapture(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    public static boolean isAvailable() {
        return GStreamerCaptureJNI.isAvailable();
    }

    public static GStreamerCapture open(String pipeline) {
        return open(pipeline, GStreamerCaptureJNI.OUTPUT_AUTO);
    }

    /**
     * Opens a capture pipeline.
     *
     * @param pipeline The GStreamer launch string. It must contain {@code appsink name=pvappsink}.
     * @param outputMode One of the {@code GStreamerCaptureJNI.OUTPUT_*} values. It selects the image
     *     layout for compressed ({@code image/jpeg}) appsink buffers.
     */
    public static GStreamerCapture open(String pipeline, int outputMode) {
        if (!isAvailable()) {
            return null;
        }

        long ptr;
        if (GStreamerCaptureJNI.hasCapability(GStreamerCaptureJNI.CAP_OPTIONS)) {
            ptr = GStreamerCaptureJNI.createWithOptions(pipeline, outputMode);
        } else {
            ptr = GStreamerCaptureJNI.create(pipeline);
        }
        return ptr == 0 ? null : new GStreamerCapture(ptr);
    }

    public boolean isOpen() {
        return nativePtr != 0;
    }

    public GStreamerFrame grab(long timeoutMillis) {
        if (nativePtr == 0) {
            return null;
        }
        return GStreamerFrame.fromNative(GStreamerCaptureJNI.grab(nativePtr, timeoutMillis));
    }

    /** Returns the flow counters, or null if the native library does not have them. */
    public Stats getStats() {
        if (nativePtr == 0 || !GStreamerCaptureJNI.hasCapability(GStreamerCaptureJNI.CAP_STATS)) {
            return null;
        }
        return Stats.fromNative(GStreamerCaptureJNI.getStats(nativePtr));
    }

    @Override
    public void close() {
        if (nativePtr != 0) {
            GStreamerCaptureJNI.destroy(nativePtr);
            nativePtr = 0;
        }
    }
}
