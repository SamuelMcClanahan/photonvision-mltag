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

package org.photonvision.vision.pipeline;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.opencv.core.Size;
import org.photonvision.common.LoadJNI;
import org.photonvision.common.configuration.ConfigManager;
import org.photonvision.common.util.TestUtils;
import org.photonvision.vision.apriltag.AprilTagFamily;
import org.photonvision.vision.calibration.CameraCalibrationCoefficients;
import org.photonvision.vision.calibration.CameraLensModel;
import org.photonvision.vision.calibration.JsonMatOfDouble;
import org.photonvision.vision.camera.QuirkyCamera;
import org.photonvision.vision.frame.provider.FileFrameProvider;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.target.TargetModel;
import org.photonvision.vision.target.TrackedTarget;
import org.wpilib.math.geometry.Transform3d;

/**
 * The poses of several tags are estimated on several threads. These tests prove that each pose is
 * exactly the pose of the calculation on one thread.
 */
public class AprilTagParallelPoseTest {
    @BeforeAll
    public static void init() {
        LoadJNI.loadLibraries();
        ConfigManager.getInstance().load();
    }

    /** A pinhole calibration with lens distortion for the 1280x720 field pictures. */
    private static CameraCalibrationCoefficients makeCalibration() {
        return new CameraCalibrationCoefficients(
                new Size(1280, 720),
                new JsonMatOfDouble(3, 3, new double[] {900, 0, 640, 0, 900, 360, 0, 0, 1}),
                new JsonMatOfDouble(1, 8, new double[] {0.04, -0.07, 0.001, -0.001, 0.01, 0, 0, 0}),
                new double[0],
                List.of(),
                new Size(0, 0),
                0,
                CameraLensModel.LENSMODEL_OPENCV);
    }

    private static AprilTagPipeline makePipeline(int threads) {
        var pipeline = new AprilTagPipeline();
        var settings = pipeline.getSettings();
        settings.useMLDetection = false;
        // 1 thread: all poses on the vision thread. 4 threads: one group of tags for each thread.
        settings.threads = threads;
        settings.solvePNPEnabled = true;
        settings.doMultiTarget = false;
        settings.doSingleTargetAlways = false;
        settings.numIterations = 40;
        settings.tagFamily = AprilTagFamily.kTag36h11;
        settings.targetModel = TargetModel.kAprilTag6p5in_36h11;
        return pipeline;
    }

    private record TagPose(int id, Transform3d best, Transform3d alternate, double ambiguity) {}

    private static List<TagPose> run(AprilTagPipeline pipeline, Path image) {
        return run(pipeline, image, makeCalibration());
    }

    private static List<TagPose> run(
            AprilTagPipeline pipeline, Path image, CameraCalibrationCoefficients calibration) {
        var provider = new FileFrameProvider(image, 70.0, 1000, calibration);
        provider.requestFrameThresholdType(pipeline.getThresholdType());
        CVPipelineResult result = pipeline.run(provider.get(), QuirkyCamera.DefaultCamera);
        List<TagPose> poses = new ArrayList<>();
        for (TrackedTarget target : result.targets) {
            poses.add(
                    new TagPose(
                            target.getFiducialId(),
                            target.getBestCameraToTarget3d(),
                            target.getAltCameraToTarget3d(),
                            target.getPoseAmbiguity()));
        }
        result.release();
        provider.release();
        return poses;
    }

    @Test
    public void parallelPosesAreEqualToSerialPoses() {
        Path folder = TestUtils.getTestImagesPath(false).resolve("2024");
        var serial = makePipeline(1);
        var parallel = makePipeline(4);
        int imagesWithSeveralTags = 0;
        int tags = 0;

        for (String name :
                List.of(
                        "GeneralField1.jpg",
                        "GeneralField2.jpg",
                        "GeneralField3.jpg",
                        "GeneralField4.jpg",
                        "GeneralField5.jpg",
                        "SpeakerCenter_63in.jpg",
                        "SpeakerCenter_143in.jpg")) {
            Path image = folder.resolve(name);
            List<TagPose> expected = run(serial, image);
            if (expected.size() < 2) {
                continue;
            }
            imagesWithSeveralTags++;
            tags += expected.size();

            // Several runs: the thread timing must not change a pose or the order of the targets.
            for (int repeat = 0; repeat < 5; repeat++) {
                List<TagPose> actual = run(parallel, image);
                assertEquals(expected.size(), actual.size(), name + ": number of targets");
                for (int i = 0; i < expected.size(); i++) {
                    assertEquals(expected.get(i).id(), actual.get(i).id(), name + ": target order");
                    // Exact equality: the same function gets the same input for each tag.
                    assertEquals(expected.get(i).best(), actual.get(i).best(), name + ": best pose");
                    assertEquals(
                            expected.get(i).alternate(), actual.get(i).alternate(), name + ": alternate pose");
                    assertEquals(
                            expected.get(i).ambiguity(), actual.get(i).ambiguity(), 0.0, name + ": ambiguity");
                }
            }
        }

        serial.release();
        parallel.release();
        assertTrue(imagesWithSeveralTags >= 2, "The test needs pictures that contain several tags");
        assertTrue(tags >= 5, "The test needs several tags");
    }

    @Test
    public void frameWithoutCalibrationAfterACalibratedFrame() {
        // The user changes to a resolution without calibration. The UI turns off 3D mode with a
        // second message, so some frames have 3D mode on and no calibration.
        Path image = TestUtils.getTestImagesPath(false).resolve("2024").resolve("GeneralField1.jpg");
        var serial = makePipeline(1);
        var parallel = makePipeline(4);
        run(serial, image);
        run(parallel, image);

        List<TagPose> expected = run(serial, image, null);
        List<TagPose> actual = assertDoesNotThrow(() -> run(parallel, image, null));
        serial.release();
        parallel.release();

        assertTrue(expected.size() >= 2, "The test needs a picture with several tags");
        assertEquals(expected.size(), actual.size(), "number of targets");
        for (int i = 0; i < expected.size(); i++) {
            assertEquals(expected.get(i).id(), actual.get(i).id(), "target order");
            assertEquals(expected.get(i).best(), actual.get(i).best(), "best pose");
            assertEquals(expected.get(i).alternate(), actual.get(i).alternate(), "alternate pose");
            assertEquals(expected.get(i).ambiguity(), actual.get(i).ambiguity(), 0.0, "ambiguity");
        }
    }
}
