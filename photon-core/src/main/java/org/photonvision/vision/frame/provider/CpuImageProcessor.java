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

package org.photonvision.vision.frame.provider;

import org.photonvision.common.util.numbers.IntegerCouple;
import org.photonvision.vision.frame.Frame;
import org.photonvision.vision.frame.FrameCaptureMetadata;
import org.photonvision.vision.frame.FrameProvider;
import org.photonvision.vision.frame.FrameStaticProperties;
import org.photonvision.vision.frame.FrameThresholdType;
import org.photonvision.vision.opencv.CVMat;
import org.photonvision.vision.opencv.ImageRotationMode;
import org.photonvision.vision.opencv.MatPool;
import org.photonvision.vision.pipe.impl.GrayscalePipe;
import org.photonvision.vision.pipe.impl.HSVPipe;
import org.photonvision.vision.pipe.impl.RotateImagePipe;

public abstract class CpuImageProcessor extends FrameProvider {
    protected static class CapturedFrame {
        CVMat colorImage;
        FrameStaticProperties staticProps;
        FrameCaptureMetadata captureMetadata;

        public CapturedFrame(
                CVMat colorImage, FrameStaticProperties staticProps, long captureTimestampNanos) {
            this(colorImage, staticProps, FrameCaptureMetadata.unknown(captureTimestampNanos));
        }

        public CapturedFrame(
                CVMat colorImage, FrameStaticProperties staticProps, FrameCaptureMetadata captureMetadata) {
            this.colorImage = colorImage;
            this.staticProps = staticProps;
            this.captureMetadata = captureMetadata;
        }

        public long captureTimestampNanos() {
            return captureMetadata != null ? captureMetadata.captureTimestampNanos() : 0;
        }
    }

    private final HSVPipe m_hsvPipe = new HSVPipe();
    private final RotateImagePipe m_rImagePipe = new RotateImagePipe();
    private final GrayscalePipe m_grayPipe = new GrayscalePipe();
    private final MatPool m_outputMatPool = new MatPool();
    FrameThresholdType m_processType;
    boolean m_blockForFrames = true;
    private boolean m_copyInput = true;
    private boolean m_copyOutput = true;
    private ImageRotationMode m_rotationMode = ImageRotationMode.DEG_0;
    private long m_lastEmptyInputPrintNanos = 0;

    private final Object m_mutex = new Object();

    abstract CapturedFrame getInputMat();

    public CpuImageProcessor() {
        m_hsvPipe.setParams(
                new HSVPipe.HSVParams(
                        new IntegerCouple(0, 180),
                        new IntegerCouple(0, 255),
                        new IntegerCouple(0, 255),
                        false));
    }

    @Override
    public final Frame get() {
        var input = getInputMat();

        m_rImagePipe.run(input.colorImage.getMat());

        CVMat outputMat = null;
        if (!input.colorImage.getMat().empty()) {
            if (m_processType == FrameThresholdType.HSV) {
                outputMat = m_outputMatPool.take();
                m_hsvPipe.processInto(input.colorImage.getMat(), outputMat.getMat());
            } else if (m_processType == FrameThresholdType.GREYSCALE) {
                if (input.colorImage.getMat().channels() == 1 && !m_copyInput) {
                    outputMat = new CVMat(input.colorImage.getMat(), null, false);
                } else {
                    outputMat = m_outputMatPool.take();
                    m_grayPipe.processInto(input.colorImage.getMat(), outputMat.getMat());
                }
            } else {
                outputMat = new CVMat();
            }

            ++sequenceID;
        } else {
            long now = System.nanoTime();
            if (now - m_lastEmptyInputPrintNanos >= 1_000_000_000L) {
                System.out.println("Input was empty!");
                m_lastEmptyInputPrintNanos = now;
            }
            outputMat = new CVMat();
        }

        return new Frame(
                sequenceID,
                input.colorImage,
                outputMat,
                m_processType,
                input.captureMetadata,
                input.staticProps != null
                        ? input.staticProps.rotate(m_rImagePipe.getParams().rotation())
                        : input.staticProps);
    }

    @Override
    public void requestFrameThresholdType(FrameThresholdType type) {
        synchronized (m_mutex) {
            if (this.m_processType == type) {
                return;
            }
            this.m_processType = type;
        }
    }

    @Override
    public void requestFrameRotation(ImageRotationMode rotationMode) {
        synchronized (m_mutex) {
            if (this.m_rotationMode == rotationMode) {
                return;
            }
            this.m_rotationMode = rotationMode;
            m_rImagePipe.setParams(new RotateImagePipe.RotateImageParams(rotationMode));
        }
    }

    /** Ask the camera to rotate frames it outputs */
    public void requestHsvSettings(HSVPipe.HSVParams params) {
        synchronized (m_mutex) {
            m_hsvPipe.setParams(params);
        }
    }

    @Override
    public void requestFrameCopies(boolean copyInput, boolean copyOutput) {
        synchronized (m_mutex) {
            m_copyInput = copyInput;
            m_copyOutput = copyOutput;
        }
    }

    @Override
    public void requestBlockForFrames(boolean blockForFrames) {
        synchronized (m_mutex) {
            this.m_blockForFrames = blockForFrames;
        }
    }

    protected void releaseProcessorResources() {
        m_outputMatPool.release();
    }
}
