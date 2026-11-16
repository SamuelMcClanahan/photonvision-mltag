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
    private long nativePtr;

    private GStreamerCapture(long nativePtr) {
        this.nativePtr = nativePtr;
    }

    public static boolean isAvailable() {
        return GStreamerCaptureJNI.isAvailable();
    }

    public static GStreamerCapture open(String pipeline) {
        if (!isAvailable()) {
            return null;
        }

        long ptr = GStreamerCaptureJNI.create(pipeline);
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

    @Override
    public void close() {
        if (nativePtr != 0) {
            GStreamerCaptureJNI.destroy(nativePtr);
            nativePtr = 0;
        }
    }
}
