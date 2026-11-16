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

package org.photonvision.vision.opencv;

import org.opencv.core.Core;
import org.photonvision.common.util.EnvUtils;

/**
 * Optional limit for the internal thread pool of OpenCV.
 *
 * <p>OpenCV divides one image operation (resize, color conversion) between all processors. The ML
 * AprilTag pipeline does its parallel work itself: one JPEG decode thread for each camera, and
 * separate tags on separate threads. The OpenCV threads then compete with these threads, and each
 * small OpenCV operation uses more processor time than it does on one thread.
 *
 * <p>Set PV_OPENCV_THREADS=1 to do each OpenCV operation on the thread that calls it. If the
 * variable is not set, the OpenCV default stays. The image results do not depend on this value.
 */
public final class OpenCVThreadSetting {
    private static boolean applied = false;

    private OpenCVThreadSetting() {}

    /** Applies PV_OPENCV_THREADS one time. The OpenCV native library must be loaded. */
    public static synchronized void apply() {
        if (applied) {
            return;
        }
        applied = true;

        int threads = EnvUtils.integer("PV_OPENCV_THREADS", -1);
        if (threads >= 0) {
            Core.setNumThreads(threads);
        }
    }
}
