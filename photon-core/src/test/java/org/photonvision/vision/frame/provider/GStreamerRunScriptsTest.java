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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/** Runs the two MLTag run scripts with a stub java, so no PhotonVision starts. */
public class GStreamerRunScriptsTest {
    private static final String ORANGE_PI = "orangepi5-run-photonvision-mltag-gstreamer.sh";
    private static final String RUBIK = "rubik-run-photonvision-mltag-gstreamer.sh";

    @TempDir Path tempDir;

    private record Result(int exitCode, String output) {}

    private static Path script(String name) {
        for (Path dir : List.of(Path.of("..", "tools"), Path.of("tools"))) {
            Path path = dir.resolve(name).toAbsolutePath().normalize();
            if (Files.isRegularFile(path)) {
                return path;
            }
        }
        assumeTrue(false, "Script not found: " + name);
        return null;
    }

    private static String homeVariable(String name) {
        return name.equals(ORANGE_PI) ? "PV_ORANGEPI_HOME" : "PV_RUBIK_HOME";
    }

    /** Makes a directory with a 'java' stub that prints its arguments. */
    private Path stubJavaDir() throws IOException {
        Path bin = Files.createDirectories(tempDir.resolve("bin"));
        Path java = bin.resolve("java");
        Files.writeString(java, "#!/usr/bin/env bash\necho \"STUB_JAVA_STARTED $*\"\n");
        Files.setPosixFilePermissions(java, PosixFilePermissions.fromString("rwxr-xr-x"));
        return bin;
    }

    private Result run(String name, Map<String, String> env, List<String> removedEnv)
            throws Exception {
        assumeTrue(Files.isExecutable(Path.of("/bin/bash")), "bash is not installed");
        var builder = new ProcessBuilder("/bin/bash", script(name).toString());
        builder.redirectErrorStream(true);
        builder.environment().putAll(env);
        removedEnv.forEach(builder.environment()::remove);
        Process process = builder.start();
        String output = new String(process.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
        assertTrue(process.waitFor(30, TimeUnit.SECONDS), "The script did not end");
        return new Result(process.exitValue(), output);
    }

    private Map<String, String> baseEnv(String name, Path library) throws IOException {
        Path home = Files.createDirectories(tempDir.resolve("home"));
        return Map.of(
                homeVariable(name),
                home.toString(),
                "PV_GSTREAMER_CAPTURE_JNI_LIB",
                library.toString(),
                "PV_PHOTONVISION_JAR",
                tempDir.resolve("dummy.jar").toString(),
                "PATH",
                stubJavaDir() + ":" + System.getenv("PATH"));
    }

    @ParameterizedTest
    @ValueSource(strings = {ORANGE_PI, RUBIK})
    public void missingLibraryStopsWithAClearMessage(String name) throws Exception {
        Path missing = tempDir.resolve("no-such-library.so");
        var result = run(name, baseEnv(name, missing), List.of());
        assertEquals(1, result.exitCode(), result.output());
        assertTrue(result.output().contains("Capture library not found: " + missing), result.output());
        assertFalse(result.output().contains("STUB_JAVA_STARTED"), result.output());
    }

    @ParameterizedTest
    @ValueSource(strings = {ORANGE_PI, RUBIK})
    public void refusedCpuListStartsJavaWithoutTaskset(String name) throws Exception {
        Path library = Files.writeString(tempDir.resolve("lib.so"), "");
        var env = new java.util.HashMap<>(baseEnv(name, library));
        env.put("PV_CPU_AFFINITY", "999");
        var result = run(name, env, List.of());
        assertEquals(0, result.exitCode(), result.output());
        assertTrue(result.output().contains("STUB_JAVA_STARTED -jar"), result.output());
    }

    @ParameterizedTest
    @ValueSource(strings = {ORANGE_PI})
    public void noHomeVariableGivesTheLibraryMessage(String name) throws Exception {
        // As a systemd service without User= the script has no HOME.
        Path home = Files.createDirectories(tempDir.resolve("home"));
        var env =
                Map.of(
                        homeVariable(name),
                        home.toString(),
                        "PV_PHOTONVISION_JAR",
                        tempDir.resolve("dummy.jar").toString(),
                        "PATH",
                        stubJavaDir() + ":" + System.getenv("PATH"));
        var result = run(name, env, List.of("HOME", "PV_GSTREAMER_CAPTURE_JNI_LIB"));
        assertFalse(result.output().contains("unbound variable"), result.output());
        if (result.exitCode() != 0) {
            assertTrue(
                    result
                            .output()
                            .contains("Capture library not found: /root/pv-gstreamer-capture-jni/build/"),
                    result.output());
        }
    }

    @ParameterizedTest
    @ValueSource(strings = {ORANGE_PI, RUBIK})
    public void missingDirectoryIsNamed(String name) throws Exception {
        Path library = Files.writeString(tempDir.resolve("lib.so"), "");
        var env = new java.util.HashMap<>(baseEnv(name, library));
        Path missing = tempDir.resolve("no-such-dir");
        env.put(homeVariable(name), missing.toString());
        var result = run(name, env, List.of());
        assertEquals(1, result.exitCode(), result.output());
        assertTrue(
                result.output().contains("Cannot enter the PhotonVision directory: " + missing),
                result.output());
    }
}
