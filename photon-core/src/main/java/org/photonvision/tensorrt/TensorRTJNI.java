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

package org.photonvision.tensorrt;

/** JNI entrypoints for the Jetson TensorRT object detector backend. */
public final class TensorRTJNI {
    public static final String JNI_LIB_ENV = "PV_TENSORRT_JNI_LIB";
    private static final boolean loaded = load();

    private TensorRTJNI() {}

    private static boolean load() {
        String explicitPath = System.getenv(JNI_LIB_ENV);
        try {
            if (explicitPath != null && !explicitPath.isBlank()) {
                System.load(explicitPath);
            } else {
                System.loadLibrary("pv_tensorrt_jni");
            }
            return true;
        } catch (UnsatisfiedLinkError | SecurityException e) {
            return false;
        }
    }

    public static boolean isAvailable() {
        return loaded;
    }

    public static native long create(String enginePath, int width, int height, int classes);

    public static native void destroy(long ptr);

    /**
     * Runs detection against a letterboxed OpenCV Mat.
     *
     * @return flattened detections as cx, cy, width, height, classId, confidence.
     */
    public static native double[] detect(
            long ptr, long matNativeObjAddr, double nmsThreshold, double confidenceThreshold);

    /**
     * Runs detection against an original OpenCV Mat. The JNI layer performs the model letterbox step
     * and returns detections already scaled back into the original image coordinate space.
     *
     * @return flattened detections as cx, cy, width, height, classId, confidence.
     */
    public static native double[] detectOriginal(
            long ptr, long matNativeObjAddr, double nmsThreshold, double confidenceThreshold);

    /**
     * Runs detection against an NVMM/RGBA GStreamer frame held by the capture JNI layer. The native
     * layer returns detections scaled back into original image coordinates.
     */
    public static native double[] detectGStreamerFrame(
            long ptr, long nativeFrameHandle, double nmsThreshold, double confidenceThreshold);
}
