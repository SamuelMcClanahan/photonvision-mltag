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

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.util.math.MathUtils;
import org.photonvision.vision.frame.FrameCaptureMetadata;

public class GStreamerFrameTest {
    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
    }

    /** A frame as the native library gives it: pull time and age on the monotonic clock. */
    private static GStreamerFrame frame(long nativePullNanos, long ageNanos) {
        return new GStreamerFrame(
                1, 1, 16, 16, 1, ageNanos, -1, -1, -1, nativePullNanos, 7, -1, -1, -1);
    }

    private static long captureNanos(GStreamerFrame frame) {
        return frame
                .toCaptureMetadata(FrameCaptureMetadata.SOURCE_GSTREAMER_JNI)
                .captureTimestampNanos();
    }

    @Test
    public void captureTimeDoesNotDependOnWhenJavaReadsTheFrame() throws InterruptedException {
        long ageNanos = 40_000_000L;
        GStreamerFrame frame = frame(System.nanoTime(), ageNanos);

        long first = captureNanos(frame);
        long wpiNow = MathUtils.wpiNanoTime();
        assertEquals(wpiNow - ageNanos, first, 2_000_000L, "The capture time is not now minus the age");

        // A JVM pause after grab() returns is like a late call. The old computation (the Java clock
        // at the call minus the age) gave a capture time 30 ms newer here.
        Thread.sleep(30);
        long second = captureNanos(frame);
        assertEquals(first, second, 1_000_000L, "A late call changed the capture time");
    }

    @Test
    public void unknownAgeUsesTheNativePullTime() throws InterruptedException {
        GStreamerFrame frame = frame(System.nanoTime(), -1);
        long first = captureNanos(frame);
        assertEquals(MathUtils.wpiNanoTime(), first, 2_000_000L);
        Thread.sleep(30);
        assertEquals(first, captureNanos(frame), 1_000_000L);
    }
}
