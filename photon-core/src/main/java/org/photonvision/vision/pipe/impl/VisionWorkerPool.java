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

package org.photonvision.vision.pipe.impl;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.photonvision.common.util.EnvUtils;

/**
 * One pool of worker threads for the short parallel steps of the vision pipelines (tag decode of
 * separate regions, pose estimation of separate tags).
 *
 * <p>All cameras share this pool. With two cameras, two private pools would start more threads than
 * the processor has cores. The thread that gives work to the pool always does a part of the work
 * itself, so the pool has one thread less than the number of processors.
 */
public final class VisionWorkerPool {
    /** Hard limit for the number of parallel groups that one pipe uses. */
    public static final int MAX_GROUPS = 8;

    /** Number of pool threads. Set PV_MLTAG_ROI_POOL_THREADS to change it. */
    public static final int THREADS =
            Math.max(
                    1,
                    EnvUtils.integer(
                            "PV_MLTAG_ROI_POOL_THREADS",
                            Math.min(MAX_GROUPS, Runtime.getRuntime().availableProcessors()) - 1));

    private static final AtomicInteger threadNumber = new AtomicInteger();

    public static final ExecutorService EXECUTOR =
            Executors.newFixedThreadPool(
                    THREADS,
                    runnable -> {
                        Thread thread =
                                new Thread(runnable, "AprilTag-ROI-Decode-" + threadNumber.incrementAndGet());
                        thread.setDaemon(true);
                        return thread;
                    });

    /**
     * Use fewer groups when more than one pipeline uses the pool. Set
     * PV_MLTAG_ROI_POOL_ADAPTIVE=false to give each pipeline the full number of groups.
     */
    private static final boolean ADAPTIVE = EnvUtils.flag("PV_MLTAG_ROI_POOL_ADAPTIVE", true);

    private static final long ACTIVE_NANOS = 1_000_000_000L;

    // The threads that asked for groups a short time ago. One thread does this work for one camera.
    private static final ConcurrentHashMap<Thread, Long> activeCallers = new ConcurrentHashMap<>();
    private static volatile int activeCallerCount = 1;
    private static volatile long activeCallerCountNanos = 0;

    private VisionWorkerPool() {}

    /**
     * Returns the number of groups that the calling thread can run at the same time for a requested
     * count: the calling thread and the pool threads.
     *
     * <p>Each camera that uses the pool also has a capture thread that decodes its frames. With one
     * camera, the parallel work makes a frame complete earlier. With more cameras than the processor
     * has free cores, the groups of the cameras would only wait for each other, which makes the frame
     * times irregular. The limit is: (processors - cameras) / cameras, and 1 or more.
     */
    public static int maxGroups(int requested) {
        int limit = Math.min(MAX_GROUPS, THREADS + 1);
        if (ADAPTIVE) {
            int cameras = countActiveCallers();
            int processors = Runtime.getRuntime().availableProcessors();
            limit = Math.min(limit, Math.max(1, (processors - cameras) / cameras));
        }
        return Math.max(1, Math.min(requested, limit));
    }

    private static int countActiveCallers() {
        long now = System.nanoTime();
        activeCallers.put(Thread.currentThread(), now);
        if (now - activeCallerCountNanos > ACTIVE_NANOS / 10) {
            activeCallers.values().removeIf(last -> now - last > ACTIVE_NANOS);
            activeCallerCount = Math.max(1, activeCallers.size());
            activeCallerCountNanos = now;
        }
        return activeCallerCount;
    }
}
