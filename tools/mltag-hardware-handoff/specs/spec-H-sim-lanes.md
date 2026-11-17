# Specification H: lane count setting in the end-to-end simulation

File that you can edit:
- photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java
Read the class comment and the function runBoard and the record Board first.

## Background
The simulation has a board profile (record Board) with the field `lanes`: the value of PV_MLTAG_INFERENCE_LANES in the run script of the board (Orange Pi 5: 2, Rubik Pi 3: 1). Each camera pipeline has this number of inference lanes. A measurement showed that with one camera, two lanes can be the limit when the NPU and the CPU are slow. The lead engineer must measure 3 lanes without a code change for each run.

## Decisions of the lead engineer
- H1: new setting PV_MLTAG_SIM_LANES (an integer from 1 to 4). When it is set, it replaces the lane count of the board profile for all scenarios of the run. When it is not set, the behavior is as now. A value outside 1 to 4 or a value that is not a number must stop the run with a clear assertion message.
- H2: the report line of each scenario (function report) must print `lanes=N` directly after the `npuMs=...` value. The settings line at the start of a run must also print the value of PV_MLTAG_SIM_LANES or `(not set)`.
- H3: add two short sentences about the setting to the class comment, next to the text about PV_MLTAG_SIM_INFERENCE_MS.
- Do not change how FPS, p99, latency or the reference comparison work. Do not change the frame set.

## Check
- Compile: `--tests "*MLTagEndToEndSimulation*"` without PV_MLTAG_SIM (the simulation tests are skipped, the tests that need no switch must pass).
- One short run with 3 lanes (permitted; about 2 minutes):
  `PV_MLTAG_SIM=1 PV_MLTAG_SIM_LANES=3 PV_MLTAG_SIM_MODE=work PV_MLTAG_SIM_CAMERAS=1,2 PV_MLTAG_SIM_SECONDS=10 PV_MLTAG_SIM_SLOWDOWNS=1.7 PV_MLTAG_SIM_SCENE_FRAMES=6 PV_OPENCV_THREADS=1 PV_GSTREAMER_CAPTURE_RT_PRIORITY=10 PV_GSTREAMER_CAPTURE_JNI_LIB=<scratch>/wt-capbuild/libpv_gstreamer_capture_jni.so ./gradlew :photon-core:cleanTest :photon-core:test --tests "*MLTagEndToEndSimulation.orangePi5" --console=plain -x spotlessApply -Porg.gradle.java.installations.paths=/usr/lib/jvm/java-25-openjdk-amd64`
  Each scenario line must show `lanes=3`, `equalToReference=100.00%` and `orderAndTimeErrors=0`. Put the lines that start with `[sim]` and contain `Orange Pi` into your report.
- One run with `PV_MLTAG_SIM_LANES=7` must fail with the assertion message.

## Rules for the implementer
- Work in <scratch>/wt in this directory. It is a separate git work tree (branch wt-impl) of the project. Edit only the file that this specification names. Do not commit, do not push, do not run spotless. The workflow of the lead engineer commits after the check.
- Build and test command for Java (JDK 25 is necessary, the option gives its path):
  `cd <scratch>/wt && ./gradlew :photon-core:test --tests "<TestClassPattern>" --console=plain -x spotlessApply -Porg.gradle.java.installations.paths=/usr/lib/jvm/java-25-openjdk-amd64`
  Only one gradle command can run at a time in one directory.
- Write comments in short simple English sentences that say WHY. No em dashes. Follow the style of the comments that are in the file.
- Final report (plain text): for each item: done or not done, the functions changed, and the last lines of the test output.
