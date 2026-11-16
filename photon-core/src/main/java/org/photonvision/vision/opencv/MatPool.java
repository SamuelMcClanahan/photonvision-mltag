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

import java.util.ArrayDeque;
import org.opencv.core.Mat;

/** Small bounded pool for reusable OpenCV Mat buffers owned by one frame provider. */
public final class MatPool implements Releasable {
    private static final int DEFAULT_MAX_RETAINED = 8;

    private final ArrayDeque<Mat> pool = new ArrayDeque<>();
    private final int maxRetained;
    private boolean released = false;

    public MatPool() {
        this(DEFAULT_MAX_RETAINED);
    }

    public MatPool(int maxRetained) {
        this.maxRetained = Math.max(0, maxRetained);
    }

    public CVMat take() {
        Mat mat;
        synchronized (this) {
            if (released || pool.isEmpty()) {
                mat = new Mat();
            } else {
                mat = pool.removeFirst();
            }
        }

        return new CVMat(mat, () -> recycle(mat), false);
    }

    private void recycle(Mat mat) {
        if (mat == null) {
            return;
        }

        synchronized (this) {
            if (!released && pool.size() < maxRetained) {
                pool.addFirst(mat);
                return;
            }
        }

        mat.release();
    }

    @Override
    public synchronized void release() {
        released = true;
        while (!pool.isEmpty()) {
            pool.removeFirst().release();
        }
    }
}
