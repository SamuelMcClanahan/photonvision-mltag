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

import java.lang.ref.PhantomReference;
import java.lang.ref.ReferenceQueue;
import java.util.Collections;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import org.opencv.core.Mat;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.EnvUtils;
import org.wpilib.util.RawFrame;

public class CVMat implements Releasable {
    private static final Logger logger = new Logger(CVMat.class, LogGroup.General);
    private static final AtomicInteger matIdCounter = new AtomicInteger(0);

    // All mats that have not yet been released(). these may still need to be GCed
    private static final Set<MatTracker> allMats =
            Collections.newSetFromMap(new ConcurrentHashMap<>());
    private static final ReferenceQueue<CVMat> refQueue = new ReferenceQueue<>();
    private static final boolean TRACK_ALLOCATIONS = parseTrackingEnabled();

    private static boolean shouldPrint;

    private Mat mat;
    private RawFrame backingFrame;
    private final Runnable releaseCallback;
    private final boolean releaseMat;
    private final long nativeFrameHandle;
    public final int matId;
    private final MatTracker tracker;
    private volatile boolean released = false;

    /** Track a single CVMat instance using a PhantomReference */
    private static class MatTracker extends PhantomReference<CVMat> {
        final int id;
        final long nativePtr;
        final String allocTrace;
        volatile boolean explicitlyReleased = false;

        MatTracker(CVMat cvmat, int id, ReferenceQueue<CVMat> queue) {
            super(cvmat, queue);
            this.id = id;
            this.nativePtr = cvmat.mat.nativeObj;
            this.allocTrace = shouldPrint ? getStackTrace() : "";
        }

        private static String getStackTrace() {
            var trace = Thread.currentThread().getStackTrace();
            final int SKIP = 4; // Skip getStackTrace, <init>, CVMat.<init>, caller
            var sb = new StringBuilder();
            for (int i = SKIP; i < Math.min(trace.length, SKIP + 10); i++) {
                sb.append("\n\t").append(trace[i]);
            }
            return sb.toString();
        }
    }

    public CVMat() {
        this(new Mat());
    }

    public CVMat(Mat mat) {
        this(mat, null);
    }

    public CVMat(Mat mat, RawFrame frame) {
        this(mat, frame, null, true);
    }

    public CVMat(Mat mat, Runnable releaseCallback, boolean releaseMat) {
        this(mat, null, releaseCallback, releaseMat);
    }

    public CVMat(Mat mat, Runnable releaseCallback, boolean releaseMat, long nativeFrameHandle) {
        this(mat, null, releaseCallback, releaseMat, nativeFrameHandle);
    }

    public CVMat(Mat mat, RawFrame frame, Runnable releaseCallback, boolean releaseMat) {
        this(mat, frame, releaseCallback, releaseMat, 0);
    }

    public CVMat(
            Mat mat,
            RawFrame frame,
            Runnable releaseCallback,
            boolean releaseMat,
            long nativeFrameHandle) {
        this.mat = mat;
        this.backingFrame = frame;
        this.releaseCallback = releaseCallback;
        this.releaseMat = releaseMat;
        this.nativeFrameHandle = nativeFrameHandle;
        this.matId = matIdCounter.incrementAndGet();
        this.tracker = TRACK_ALLOCATIONS ? new MatTracker(this, matId, refQueue) : null;

        if (TRACK_ALLOCATIONS) {
            allMats.add(tracker);
        }

        if (shouldPrint && tracker != null) {
            logger.trace("CVMat" + matId + " allocated - count: " + allMats.size() + tracker.allocTrace);
        }
    }

    public void copyFrom(CVMat srcMat) {
        copyFrom(srcMat.getMat());
    }

    public void copyFrom(Mat srcMat) {
        srcMat.copyTo(mat);
    }

    @Override
    public void release() {
        synchronized (this) {
            if (released) {
                if (shouldPrint) {
                    logger.error("CVMat" + matId + " already released (ignored)");
                }
                return;
            }
            released = true;
        }

        if (tracker != null) {
            tracker.explicitlyReleased = true;
        }

        // Free RawFrames exactly ONCE
        if (backingFrame != null) {
            try {
                backingFrame.close();
                backingFrame = null;
            } catch (Exception e) {
                logger.error("Error closing RawFrame for CVMat" + matId, e);
            }
        }

        try {
            if (releaseMat && mat != null) {
                mat.release();
                mat = null;
            } else if (mat != null) {
                mat = null;
            } else {
                logger.error("Mat was already null, this is a no-op");
            }
        } catch (Exception e) {
            logger.error("Error releasing Mat for CVMat" + matId, e);
        }

        if (releaseCallback != null) {
            try {
                releaseCallback.run();
            } catch (Exception e) {
                logger.error("Error running release callback for CVMat" + matId, e);
            }
        }

        // write down it's freed
        if (tracker != null) {
            allMats.remove(tracker);
        }

        if (shouldPrint && tracker != null) {
            logger.trace("CVMat" + matId + " released - count: " + allMats.size());
        }
    }

    public Mat getMat() {
        if (released) {
            throw new IllegalStateException("CVMat" + matId + " has been released!");
        }
        return mat;
    }

    public boolean isReleased() {
        return released;
    }

    public long getNativeFrameHandle() {
        return nativeFrameHandle;
    }

    @Override
    public String toString() {
        return "CVMat [mat="
                + mat
                + ", backingFrame="
                + backingFrame
                + ", matId="
                + matId
                + ", tracker="
                + tracker
                + ", released="
                + released
                + "]";
    }

    public static int getMatCount() {
        return allMats.size();
    }

    public static void enablePrint(boolean enabled) {
        shouldPrint = enabled;
    }

    private static boolean parseTrackingEnabled() {
        return EnvUtils.flag("PV_CVMAT_TRACKING", true);
    }

    // todo move to somewhere else
    static {
        if (TRACK_ALLOCATIONS) {
            Thread cleanupThread =
                    new Thread(
                            () -> {
                                while (true) {
                                    try {
                                        MatTracker ref = (MatTracker) refQueue.remove();

                                        // Check if it was released before GC
                                        if (!ref.explicitlyReleased) {
                                            // This is a leak - remove from tracking and warn
                                            allMats.remove(ref);

                                            logger.warn(
                                                    "CVMat"
                                                            + ref.id
                                                            + " was GC'd without release()! "
                                                            + "Native memory may have leaked."
                                                            + "\nAllocated by "
                                                            + ref.allocTrace);
                                            if (ref.allocTrace != null) {
                                                logger.warn("Allocated at:" + ref.allocTrace);
                                            }
                                        }

                                        // Because we use PhantomReferences, we can't try to be nice
                                    } catch (InterruptedException e) {
                                        Thread.currentThread().interrupt();
                                        break;
                                    }
                                }
                            },
                            "CVMat-Cleanup");
            cleanupThread.setDaemon(true);
            cleanupThread.start();
        }
    }

    // Paranoia
    @Override
    @SuppressWarnings("deprecation")
    protected void finalize() throws Throwable {
        try {
            if (!released && tracker != null) {
                logger.error(
                        "CVMat"
                                + matId
                                + " finalized without release()! Leaking native memory. Allocated by "
                                + tracker.allocTrace);
                // Don't call release() here - finalization order is unpredictable
                // and backingFrame might already be finalized
            }
        } finally {
            super.finalize();
        }
    }
}
