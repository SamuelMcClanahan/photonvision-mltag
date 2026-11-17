# Specification G: stronger checks in the end-to-end simulation

File that you can edit:
- photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java
Read the class comment and the complete file first. Earlier specifications changed this file (gray camera format, JPEG factor, shared model input helper); keep those changes.

## Decisions of the lead engineer
- G1 (compare each result from the first one): implement PART 1 of the corrected method of the check. Do NOT implement the PV_MLTAG_SIM_DISTURB thread of the first method.
- G2 (poses in work mode): implement the corrected method of the check: one reference for each iteration count, always compare all 15 values.
- G3 (order and time): implement the SMALL version from the corrected method of the check (three requirements, a counter orderAndTimeErrors in the report, an assertion of 0). Do not add timing windows. Add the proof run that the corrected method names as a JUnit test that needs PV_MLTAG_SIM=1.
- G4 (CPU count and run settings): implement the corrected method of the check. The report must print PV_OPENCV_THREADS, PV_GSTREAMER_CAPTURE_RT_PRIORITY, PV_GSTREAMER_CAPTURE_JPEG_IDCT, the camera format, VisionWorkerPool.THREADS and the number of CPUs.
- G5 (test state): runScenario must restore the test state in a finally block: setTestExtraDecodePercent(0), stop the loops, release pipelines and captures, also when an assertion or an exception stops the scenario.
- G6 (NaN): compareWithReference (or its new form) must report a difference when one value is NaN and the other is not, and when both are NaN it must say so as a difference too (a NaN pose is never a valid equal result).
- Do not change how FPS, p99 and latency are computed, and do not change the frame set.
- Check: compile with `--tests "*MLTagEndToEndSimulation*"` (skipped without the switch). Then run one short simulation in each time mode to prove that the checks pass on correct code (these runs are permitted; each takes about 2 to 3 minutes):
  `PV_MLTAG_SIM=1 PV_MLTAG_SIM_MODE=work PV_MLTAG_SIM_CAMERAS=2 PV_MLTAG_SIM_SECONDS=10 PV_MLTAG_SIM_SLOWDOWNS=1.5 PV_MLTAG_SIM_SCENE_FRAMES=6 PV_OPENCV_THREADS=1 PV_GSTREAMER_CAPTURE_RT_PRIORITY=10 PV_GSTREAMER_CAPTURE_JNI_LIB=<library path from the Rules> ./gradlew :photon-core:cleanTest :photon-core:test --tests "*MLTagEndToEndSimulation.orangePi5" --console=plain -x spotlessApply -Porg.gradle.java.installations.paths=/usr/lib/jvm/java-25-openjdk-amd64`
  and the same without PV_MLTAG_SIM_MODE=work. Put the lines that start with `[sim]` into your report. Each scenario line must show equalToReference=100.00% and orderAndTimeErrors=0.

## Rules for the implementer
- Work in <scratch>/wt in this directory. It is a separate git work tree (branch wt-impl) of the project. The specifications before yours are committed there. Edit only the files that this specification names. Do not commit, do not push, do not run spotless. The workflow of the lead engineer commits after the check.
- Do not move or rename files. Do not work in any other copy of the project, except where this specification tells you to make a second work tree of the base commit.
- Build and test command for Java (JDK 25 is necessary, the option gives its path):
  `cd <scratch>/wt && ./gradlew :photon-core:test --tests "<TestClassPattern>" --console=plain -x spotlessApply -Porg.gradle.java.installations.paths=/usr/lib/jvm/java-25-openjdk-amd64`
  Only one gradle command can run at a time in one directory. If gradle says that the build directory is locked, wait and try again.
- The native capture library is built with: `cd <scratch>/wt-capbuild && make` (the build directory is configured for tools/jetson-gstreamer-capture-jni of the work tree). Tests that use the library need the environment variable `PV_GSTREAMER_CAPTURE_JNI_LIB=<scratch>/wt-capbuild/libpv_gstreamer_capture_jni.so` in front of the gradle command. A second build without the JPEG decoder must also compile: `cd <scratch>/wt-capbuild-nojpeg && make`.
- The hard requirement of the owner: no interference with accuracy. Tag IDs, tag corners, poses and capture timestamps must not get worse, and no stale or mixed-up data.
- Write comments in short simple English sentences that say WHY. No em dashes. Follow the style of the comments that are in the file.
- Each item needs a test that fails before the change and passes after it, as the item says. All tests that are in the named test classes now must pass after your work.
- If a method in this specification cannot work as written, stop on that item, keep the other items, and say exactly why in your report. Do not invent a different design.
- Final report (plain text): for each item: done or not done, the files and functions changed, the tests added, and the last lines of the test output.

## Items
### G1: Results in the 5 s warm-up are never compared, so the mode changes (SERIAL to OVERLAP, DEFER, SKIP_INPUT, depth steps, stale-frame discard) are outside the check
- Location: /home/user/photonvision-mltag/photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java:1549
- Failure scenario: The controller goes to OVERLAP after 8 calls and adds depth after 8 more calls, so all transitions occur in the first second. A fault in a transition path (wrong frame released on SKIP_INPUT, a frame released while a lane still reads it, a result for the wrong pending frame after a depth step or after discardStalePendingFrames) gives wrong tag data only for the first results. They are not compared and not counted, and the run prints 100%. The steady state of 6 to 30 s has no transition (a depth decrease needs 600 calls, a stale discard needs a 500 ms stall), so these paths are never checked under load.
- Evidence: Line 1549: Thread.sleep(5000) before measuring.set(true). Lines 1433-1453: readFrameIndex, compareWithReference and the counters are inside if (measuring.get()).
- First method from the review: Move the index read and the comparison out of the measuring block: compare each result from the first one and count in separate fields (checkedResults, checkedEqual, firstDifference); keep the timing lists inside the measuring block. Assert checkedEqual == checkedResults in runBoard. Add PV_MLTAG_SIM_DISTURB=1: a thread that during the measurement (a) each 4 s holds one SimVision loop for 700 ms (a volatile pauseNanos that the loop reads before camera.get()), which forces the stale discard and a controller reset, and (b) sets the NPU time to 0.3 x for 2 s (make SimulatedNpu.inferenceNanos volatile), which forces OVERLAP to SERIAL and back with SKIP_INPUT. Timing values of such a run are not for the report; the data checks must stay at zero errors. Proof: the log must show the lines 'ML AprilTag inference lanes: one frame at a time' and '... N frames in work' during the measurement, and checkedResults must be larger than results.
- Result of the independent check: real=True, confidence=medium. Reasoning: Could not refute. The gap exists as described, but it is a hole in the evidence, not a demonstrated product fault, so severity is low to medium.

Facts checked in /home/user/photonvision-mltag/photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java (HEAD 5404a7e, clean tree):
- runScenario starts the SimVision threads at line 1542, sleeps 5000 ms at line 1549, and sets measuring=true only at line 1568.
- In CameraLoop.run, readFrameIndex, compareWithReference, results, resultsEqualToReference and firstDifference are all inside "if (measuring.get())" (lines 1433-1453). Before that, a result goes directly to result.release() (line 1454).
- The only check active during warm-up is the static unknownFrames counter, which shows that the simulated model could read the index. It does not check the tag data of a result.
- The class Javadoc (line 110) and the comment at line 1876 say that each result of a load run is compared. That is not what the code does.

Why the transitions fall outside the window (InferenceOverlapController.java):
- SERIAL to OVERLAP needs 8 low-idle calls (CALLS_TO_START_OVERLAP), then a DEFER call for each depth step after 8 frame-period samples and 8 calls (CALLS_TO_ADD_DEPTH). At 60 x k FPS this is complete in about the first second.
- In steady state a step down needs 600 calls, or 60 calls above the limit. A stale discard needs a frame gap above 500 ms (STALE_PENDING_NANOS in AprilTagPipeline.java).
- The reviewer's "never" is a little too strong. A depth step down can fall into the window by chance (for example at 5 s + 600 calls after a JIT overshoot, or from the above-limit path in two-camera runs). Nothing guarantees it. The first SERIAL to OVERLAP change and the first DEFER calls are always before 5 s.

Existing tests do not cover it:
- AprilTagPipelineInferenceLanesTest is the only other test with a multi-lane pipeline. twoLanesGiveTheSameTagDataInTheSameOrder checks IDs and corners from call 1 across SERIAL to OVERLAP, but each frame is the same picture, so tag data of frame A in the result of frame B cannot be seen. It also has solvePNPEnabled=false.
- Its capture-time assert is equal by construction: CVPipeline.run sets the result time from result.inputAndOutputFrame.
- overlapStopsWhenTheModelBecomesFast runs the SKIP_INPUT path but does not look at the tag data of the 150 transition calls.
- No test has a frame gap above 500 ms, so no test runs discardStalePendingFrames.
- InferenceOverlapControllerTest has no frames.
- The simulation is the only test with a different picture and reference for each frame, and it starts to compare after the transitions.

What I did not find: a wrong-data defect in the hidden paths. I traced processWithInferenceLanes, discardStalePendingFrames, finishFrame, and AprilTagMLHybridPipe.submit/complete:
- A SKIP_INPUT frame is released without a submit to a lane.
- A stale discard waits for the Future before the release.
- Each result is tied to its frame through one Future.
The Rubik profile (1 lane) has no transitions and is not affected.

Faults in the proposed fix method:
1. If the counters move to new fields and the PV_MLTAG_SIM_NULL branch is not changed, the fraction is 0/0 = NaN and assertEquals(1.0, NaN, 1e-12) fails.
2. NPU time x 0.3 does not reliably stop OVERLAP. The stop needs inference average + post-inference average below 0.9 x camera period for 60 calls in sequence. At k=2, in work mode, or with two cameras, the post-inference time can keep the sum above that limit, and then nothing occurs.
3. The log-line proof cannot be asserted by the test. The stale-discard reset alone also prints "one frame at a time" and then "N frames in work", so the log does not prove that a SKIP_INPUT occurred.
4. A 4 s cycle gives at most one event of each type in the default 6 s window.
- Corrected method from the check (use THIS method unless the "Decisions" section says something different): All changes are in photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java. No product code changes.

PART 1 (necessary): compare each result from the first one.
- CameraLoop: add fields long checkedResults and long checkedEqual. Keep results (for FPS) and firstDifference.
- CameraLoop.run, after "if (result == null) continue;": move these lines out of the measuring block so they run for each result:
  - int index = readFrameIndex(result.inputAndOutputFrame.colorImage.getMat());
  - difference = index < 0 ? "a result frame has no valid index" : compareWithReference(index, tagData(result), comparePose);
  - checkedResults++; if difference == null then checkedEqual++, else set firstDifference if it is null.
- Keep inside "if (measuring.get())": pipeNanos, results++, resultSceneTypes (it uses index), resultPipeNanos, latencyNanos, resultTimesNanos, resultRunNanos, reportedFps.
- In the pipeline == null branch, increment checkedResults and checkedEqual for each frame, outside the measuring block. Without this the fraction is 0/0 = NaN in PV_MLTAG_SIM_NULL mode.
- runScenario: sum checkedResults and checkedEqual over the loops. Put both in Outcome. Calculate equalToReferenceFraction as checkedEqual / checkedResults.
- runBoard: assertEquals(outcome.checkedResults(), outcome.checkedEqual(), "A result is not equal to the reference: " + firstDifference). Also assert checkedResults > the sum of loop.results when a pipeline is present.
- Correct the comment at line 1548 and the class Javadoc: the comparison includes the warm-up.

PART 2: frame accounting in CameraLoop. This gives a proof that the transitions occurred, and a capture-order check.
- Add ArrayDeque<Frame> inWork and counters deferCalls, skipInputCalls, staleDiscardedFrames, ownFrameResults, olderFrameResults. Count them only while measuring.get() is true. Always keep inWork up to date.
- Directly after pipeline.run(...), on the SimVision thread:
  a. While inWork is not empty and inWork.peekFirst().colorImage.isReleased(): removeFirst and staleDiscardedFrames++. The pipeline released such a frame and gave no result for it.
  b. If result == null: deferCalls++, inWork.addLast(frame), continue.
  c. If result.inputAndOutputFrame == frame: ownFrameResults++. If inWork is not empty, record the difference "own frame returned while older frames are in work".
  d. Else: olderFrameResults++. If inWork.peekFirst() != result.inputAndOutputFrame, record the difference "result is not for the oldest frame in work"; else removeFirst. Then, if frame.colorImage.isReleased(): skipInputCalls++; else inWork.addLast(frame).
- Reading isReleased() here has no race. Each frame release occurs on this thread, in pipeline.run or in result.release() at the end of the iteration.
- A difference from c or d goes to firstDifference and makes the result not equal.

PART 3: PV_MLTAG_SIM_DISTURB=1 forces the transitions inside the measurement.
- CameraLoop: add volatile long pauseNanos and AtomicInteger pauseCalls. At the top of each loop iteration, before getStart: if pauseCalls.get() > 0, decrement it and sleep pauseNanos.
- runScenario: after measuring.set(true), start a thread that stops before measuring.set(false). It repeats a 3 s cycle relative to the start:
  - at +0.5 s, for each loop: pauseNanos = 100 ms, pauseCalls = 6;
  - at +2.0 s, for each loop: pauseNanos = 700 ms, pauseCalls = 1.
- Effect of the 100 ms pauses: four calls in sequence with idle above 1.5 x work. This is CALLS_TO_STOP_OVERLAP_ON_LONG_IDLE, so the controller stops OVERLAP. The pipeline gets SKIP_INPUT for each frame in work, runs SERIAL, and after 8 low-idle calls starts OVERLAP again with DEFER.
- Why 100 ms: it must be above 1.5 x (inference + post-inference time), and MAX_DEPTH x 100 ms = 400 ms must stay below the 500 ms stale limit.
- Effect of the 700 ms pause: the next frame is more than 500 ms newer than the frames in work. discardStalePendingFrames releases them and resets the controller. OVERLAP then starts again.
- Do not use the NPU x 0.3 method. It depends on the post-inference time of the computer. The pipeline gets the same SKIP_INPUT action for both causes, so the pause covers the same pipeline path.
- In this mode, print in the report line that the timing values are not valid.

How to prove it:
- Normal run: checkedEqual == checkedResults and checkedResults > results.
- DISTURB run on the Orange Pi profile (lanes > 1), for each camera loop, counted during the measurement:
  - skipInputCalls >= 1, deferCalls >= 1, staleDiscardedFrames >= 1, ownFrameResults >= 1, olderFrameResults >= 1;
  - checkedEqual == checkedResults;
  - unknownFrames == 0.
  Do not make the counter asserts for a profile with 1 lane.
- Mutation check, done by hand and not committed. In AprilTagPipeline.finishFrame, remove the last detection for the first 20 frames of a pipeline. The old simulation prints 100%. The new one must fail with firstDifference.
- Second mutation check: in processWithInferenceLanes, use pendingFrames.removeLast() in place of removeFirst() when the action is SKIP_INPUT. The DISTURB run must fail with "result is not for the oldest frame in work".

### G2: Work mode does not compare poses, and it changes the pose iterations to 52-80: equalToReference=100% gives no proof for poses in the mode that is recommended for p99
- Location: /home/user/photonvision-mltag/photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java:1538
- Failure scenario: PV_MLTAG_SIM_MODE=work (the command in the class comment). A fault in the parallel pose code that is only visible under load (for example a pose written to the wrong index when groups run on the shared pool, or a pose pipe used by two threads) changes poses but not IDs or corners. compareWithReference then checks only values 0 to 8, and the report prints equalToReference=100.00% with no sign that poses were not compared.
- Evidence: Line 1538: new CameraLoop(..., workMultiplier == 1.0) sets comparePose false in work mode. Line 914: int values = withPose ? 15 : 9. Line 1315: settings.numIterations = (int) Math.round(40 * workMultiplier), so the load run cannot be compared with the 40-iteration reference. The report line (1795-1816) does not say that the pose was skipped.
- First method from the review: Make the reference for each iteration count that a load run uses. Change prepareReference() to prepareReference(int numIterations), keep the result in a Map<Integer, List<List<double[]>>> (key = numIterations), and call it at the start of runScenario for makeSettings(workMultiplier).numIterations when the key is missing (one calm lap for each value of k, about N x 33 ms). Give the CameraLoop the list for its iteration count and always compare all 15 values (remove the withPose parameter). Proof: a unit test of compareWithReference with a difference of 1e-6 in value 12 must return a text for each mode; a work-mode run with k=1.5 must still give 100% with the pose compared.
- Result of the independent check: real=True, confidence=high. Reasoning: I could not refute it. The path is certain from the code in /home/user/photonvision-mltag/photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java (current tree, HEAD 5404a7e, no uncommitted edits).

Trace:
- Line 1492: workMultiplier = WORK_MODE ? slowdown : 1.0. The default slowdowns are 1.3, 1.5, 2.0 (line 1822), so in work mode the value is never 1.0.
- Line 1538: new CameraLoop(..., workMultiplier == 1.0) gives comparePose=false for every work-mode scenario.
- Line 1441 passes comparePose to compareWithReference. Line 914: values = withPose ? 15 : 9, so values 9 to 14 (pose x, y, z and rotation) are not read.
- Line 1315: numIterations = round(40 * workMultiplier) = 52, 60, 80. The reference (line 952) uses makeSettings(1.0) = 40. This is the reason the author gave for the skip (Javadoc of withPose, line 905, and the comment at line 1536).
- The report (lines 1795-1816), the mode line (1845-1851) and the assert message (1876-1883) do not say that the pose was not compared. The class comment (lines 109-110) says each load result "must be equal to the recorded data of its frame", and the command in the class comment (line 139) uses PV_MLTAG_SIM_MODE=work, which lines 119-123 call the better mode for p99.

So in work mode a result with a wrong pose but correct ID and corners is counted in resultsEqualToReference and the run prints equalToReference=100.00% and passes the assert at line 1879.

No other test closes the gap for the overlap path:
- AprilTagPipelineInferenceLanesTest sets solvePNPEnabled=false (line 205), so no pose is compared with lanes or the frame work thread.
- AprilTagParallelPoseTest compares parallel with serial poses only on the non-ML path, on one vision thread, with no overlap.
- The time-scaling mode of the simulation (default) does compare all 15 values under load, so poses in OVERLAP mode have some proof. This limits the severity: I put it at low to medium, not medium. The gap is only in work mode, but that is the mode the documented command uses.

Limits of the claim: the "pose fault that is visible only in work mode" is hypothetical; I found no such fault in estimateSingleTagPoses / estimatePoseGroup (AprilTagPipeline.java 619-717: each group has its own pose pipe and writes poses[index] at separate indexes). The real fault is the measurement: the printed value claims more than the check does.

The proposed fix is correct in principle. The pose is a pure function of the corners, the calibration and nIters (AprilTagPoseEstimatorPipe line 85, estimateOrthogonalIteration(corrected, nIters)), so a calm run with the same iteration count gives the same pose as the load run. The duplicate regions of work mode do not change this: deduplicateByTagId (AprilTagROIDecodePipe 694-711) keeps one of two identical detections. The fix text is not complete in three points, corrected below.
- Corrected method from the check (use THIS method unless the "Decisions" section says something different): File: photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java. Test code only.

1. Reference store. Replace the static field referenceTags (line 233) with: private static final Map<Integer, List<List<double[]>>> referenceByIterations = new HashMap<>(); (key = numIterations). Add a constant BASE_ITERATIONS = 40 and a helper iterationsFor(double workMultiplier) = (int) Math.round(BASE_ITERATIONS * workMultiplier). Use the helper in makeSettings (line 1315) too, so the two places cannot differ.

2. compareWithReference. New signature: compareWithReference(List<List<double[]>> reference, int index, List<double[]> tags). Remove the withPose parameter and always compare all 15 values. It must not read a static field. This point is missing in the proposed fix: today the function reads the static referenceTags, and prepareReference swaps that static for its stability check (lines 979-982). With a map that swap is wrong; pass the list as a parameter.

3. prepareReference(int numIterations). Returns List<List<double[]>> and puts it into the map. Inside: settings = makeSettings(1.0); settings.numIterations = numIterations; the model stays new SimulatedModel(new SimulatedNpu(1, 0, 0.0, 0), 1.0) (no duplicate regions, no extra prepare: only the iteration count changes). The stability check becomes compareWithReference(reference, index, tags) on the local list. Call GStreamerCaptureJNI.setTestExtraDecodePercent(0) at its start so the calm run never runs with the extra decode of an earlier scenario.

4. Where to call. prepareScene (line 802) calls prepareReference(BASE_ITERATIONS); the statistics at lines 822 and 860 read referenceByIterations.get(BASE_ITERATIONS) (the tag count does not depend on the iterations). In runBoard, in the loop over slowdowns and BEFORE calibrateCameraSleepMicros (line 1864), do: int iterations = iterationsFor(WORK_MODE ? slowdown : 1.0); if the map has no entry, prepareReference(iterations). Do not do it inside runScenario after setTestExtraDecodePercent and the CPU mask (lines 1493-1514); the proposed "start of runScenario" is acceptable only if it is the first statement. Cost: one calm lap for each new k (number of frames x 30 ms).

5. CameraLoop. Replace the boolean comparePose with final List<List<double[]>> reference. runScenario gives referenceByIterations.get(iterationsFor(workMultiplier)) and asserts that it is not null. Line 1441 becomes compareWithReference(reference, index, tagData(result)). Remove the comment at line 1536 and the @param withPose text.

6. Report. Add the iteration count to the mode line or to the report line (for example "poseIterations=52, pose compared"), so the output states what was compared.

Proof:
a) A new @Test in the same class without the PV_MLTAG_SIM gate: make a reference list with one frame and one tag (15 values), make a copy with value 12 changed by 1e-6, and assert that compareWithReference returns a text that contains "value 12"; assert null for an equal copy; assert a text for a change of 1e-6 in value 3 (a corner). Before the fix the pose case cannot be expressed without withPose=true; after the fix there is only one path.
b) Negative check by hand for the work mode: with PV_MLTAG_SIM_MODE=work and PV_MLTAG_SIM_SLOWDOWNS=1.5, change tagData temporarily to add 1e-6 to values[12] in the load run only (or give the CameraLoop the 40-iteration reference); the run must fail with "A result is not equal to the reference: frame N tag M value 12". Before the fix it passes with 100.00%.
c) The normal work-mode run (k = 1.3, 1.5, 2.0, one and two cameras) and the time-scaling run must both still give equalToReference=100.00% with all 15 values compared, and "The reference run is not stable" must not occur for 52, 60 and 80 iterations.

### G3: The simulation does not check capture timestamp, capture order or duplicates: a result with the wrong timestamp passes with equalToReference=100% and a BETTER latency
- Location: /home/user/photonvision-mltag/photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java:1433
- Failure scenario: Put the base line back in CVPipeline.run (result.setImageCaptureTimestampNanos(frame.timestampNanos), the timestamp of the INPUT frame of the call). In OVERLAP mode the result is for a frame that is 'depth' camera periods older, so robot code gets a pose with a timestamp 17 to 50 ms too new. The simulation still prints equalToReference=100%, and latency p50 goes DOWN by depth x 16.7 ms, so the wrong build looks better. The same is true if the pipeline returns results out of capture order (removeLast in place of removeFirst) or returns one finished frame two times (resultFps then counts the duplicate as a result).
- Evidence: CameraLoop.run lines 1433-1453: the only data check is compareWithReference(index, tagData(result), comparePose). tagData (877-900) holds ID, 4 corners, best pose only. result.getImageCaptureTimestampNanos() is used only at line 1449 to compute latency. The loop keeps no previous index and no previous timestamp. The only timestamp check in the repository is AprilTagPipelineInferenceLanesTest (one static picture, FileFrameProvider, no native capture, no load).
- First method from the review: In CameraLoop: (1) after camera.get() for a non-empty frame, read idx = readFrameIndex(frame.colorImage.getMat()) and store grabbedCaptureNanos[idx] = frame.timestampNanos (long[sceneFrames.length] for each loop). Require frame.timestampNanos > the timestamp of the last grabbed frame and 0 <= MathUtils.wpiNanoTime() - frame.timestampNanos <= 250 ms. (2) For EVERY result (not only while measuring): index = readFrameIndex(result picture); require result.getImageCaptureTimestampNanos() == grabbedCaptureNanos[index] and == result.inputAndOutputFrame.timestampNanos; require the timestamp to be strictly larger than the timestamp of the previous result (order, no duplicate); with d = floorMod(index - previousIndex, N) require 1 <= d < N/2; require d*sleepMicros*1000 - 0.25*period <= (ts - previousTs) <= d*period + 50 ms (the simulated camera cannot give d frames in less than d sleeps, so this ties the native capture time to the picture). Count violations in a new field, keep the first text, add it to Outcome, print 'orderAndTimeErrors=', and assertEquals(0, ...) in runBoard. Proof: put the checks in a small static class ResultChecker (onGrab(index, ts), onResult(index, ts) returns String) and add a JUnit test that needs no environment flag: a correct sequence gives null; a result with the timestamp of the next grabbed frame, two swapped results, and one result given two times each give an error. Also run the simulation one time with a SimulatedPipeline subclass whose run() copies the input frame timestamp into the result: the run must fail.
- Result of the independent check: real=False, confidence=medium. Reasoning: The observation is accurate, but it is a missing extra assertion in an opt-in simulation, not a fault in the delivered code. No sequence of events in the current tree gives wrong data or a wrong measurement, and each regression the reviewer names is already caught elsewhere.

**What is true in the report**
- `CameraLoop.run` (`MLTagEndToEndSimulation.java` 1433-1453) checks only `compareWithReference(index, tagData(result), comparePose)`.
- `tagData` (877-900) holds ID, 4 corners and pose only.
- `result.getImageCaptureTimestampNanos()` is used only at line 1449 for latency; the loop keeps no previous index or timestamp.

**Why the failure scenario does not hold**
1. **The current code is correct, so the measurement is correct.**
   - `CVPipeline.run` (lines 91-92) sets the timestamp from `result.inputAndOutputFrame.timestampNanos`; this is the only call of `setImageCaptureTimestampNanos` in main code.
   - `AprilTagPipeline.buildResult` (597-605) puts the same `Frame` object into the result that the detection was computed from (`PendingDetection.frame`, `finishFrame` 400-413).
   - `Frame.timestampNanos` is final and comes from the same native `FrameHolder` as the picture (`fillTimestamps`, `GStreamerCaptureJNI.cpp` 1015-1032, reads the meta of the same `GstBuffer`).
   - `pendingFrames` is strictly `addLast`/`removeFirst` (385, 393); the native slot is taken and set to null under the mutex (1808-1809), so no frame is delivered twice.

2. **Reverting the timestamp line is caught by a default test.** `AprilTagPipelineInferenceLanesTest.runAtCameraRate` (lines 293-295) asserts, for every result, that the result timestamp and sequence ID equal those of `result.inputAndOutputFrame`. Overlap is proven active in those tests (`maxRunning == 2`, `callsWithoutResult >= 1`, `assertFalse(slow.ownFrame.get(last))`), and each frame has a different timestamp. The test needs no environment flag and no native capture, so the wrong build fails the normal `photon-core` test run.

3. **A duplicate is caught by the simulation itself.** The loop calls `result.release()` at line 1454. If the same result came back during measuring, `colorImage.getMat()` at line 1437 throws `IllegalStateException` (`CVMat.getMat` 190-193). That is caught at 1456, stored in `loop.failure`, and `runScenario` throws "The vision loop failed" (1617-1619). The duplicate is not counted in `resultFps`. `twoLanesGiveTheSameTagDataInTheSameOrder` (329-335) also requires strictly increasing sequence ID and capture time.

4. **`removeLast` in place of `removeFirst` does not give out-of-order results.** The call adds its own frame last and removes it again, so every call returns its own newest frame in capture order. Older frames stay in the deque until the 500 ms stale discard. That is a throughput fault, which shows in the FPS and `ownFrame` assertions of the lanes test, not a timestamp or order fault.

5. **Mixed-up data is already covered.** The simulation reads the index from the picture of `result.inputAndOutputFrame` and compares the tag data with the reference of that picture. Together with point 1 and the lanes-test assertion, the chain data → picture → frame timestamp → result timestamp is covered.

**Why confidence is medium, not high**
The simulation alone would indeed pass a build with the timestamp line reverted. If the owner treats the simulation as the only acceptance evidence, the reviewer's point has some weight as test hardening (low severity, not high).

**Problems in the proposed fix**
The timing windows (`0..250 ms` age, `d*period + 50 ms` upper limit) would make a performance run fail on a scheduler or GC stall. The simulation loads all CPUs on purpose, so such stalls are expected.
- Corrected method from the check (use THIS method unless the "Decisions" section says something different): No fix is required. If the owner wants the check in the simulation as optional hardening, use this smaller version, which has no timing windows.

In `MLTagEndToEndSimulation.CameraLoop.run`, for every non-null result (also outside the measuring window), before `result.release()`:
1. Require `result.getImageCaptureTimestampNanos() == result.inputAndOutputFrame.timestampNanos`.
2. Require `result.getImageCaptureTimestampNanos() > previousResultCaptureNanos` (a field of the loop, start value `Long.MIN_VALUE`), then update the field. This covers order and duplicates.
3. Require `result.sequenceID == result.inputAndOutputFrame.sequenceID`.

On a violation, increase a new field `orderAndTimeErrors` and keep the first text. Add the sum to `Outcome`, print `orderAndTimeErrors=` in `report`, and add `assertEquals(0, outcome.orderAndTimeErrors())` in `runBoard`.

Do not add the proposed limits `0 <= now - ts <= 250 ms` and `(ts - previousTs) <= d*period + 50 ms`. A scheduler or GC stall would then fail a performance run with no data fault.

Proof: run the simulation one time with a `SimulatedPipeline` subclass that overrides `run()` and sets the result timestamp to the input frame timestamp. In OVERLAP mode `orderAndTimeErrors` must be greater than 0 and the run must fail.

### G4: The CPU count and the run settings are not enforced or printed in a way that stops a wrong run: on a host with more than 4 CPUs the result is optimistic with no warning; PV_MLTAG_SIM_APP_CPUS is applied too late
- Location: /home/user/photonvision-mltag/photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java:1844
- Failure scenario: The class comment asks to run the simulation on a different computer. On an 8-CPU host with two cameras VisionWorkerPool.maxGroups gives (8-2)/2 = 3 groups, so tag decode and pose run in parallel, and 8 CPUs never have the contention of the board. On the Orange Pi (taskset -c 4-7, 4 CPUs) the same call gives (4-2)/2 = 1 group (serial). The run still prints one normal report line. This host has 4 CPUs (/sys/devices/system/cpu/online = 0-3), so the present numbers have the correct group count, but only by chance. With PV_MLTAG_SIM_APP_CPUS the mask is set in runScenario, after prepareScene: the pool threads (all are created by the main thread in the reference run), the GStreamer threads of the reference run and all JVM service threads (GC, JIT) keep all CPUs, and VisionWorkerPool.THREADS was already calculated from the full count.
- Evidence: runBoard line 1844 calls prepareScene() (reference run, VisionWorkerPool class load and thread creation) before runScenario lines 1504-1514 set the mask. Line 1851 only prints the CPU count. VisionWorkerPool.java lines 39-44 (THREADS from availableProcessors at class load) and 82-90 (limit = (processors - cameras) / cameras). Run script tools/orangepi5-run-photonvision-mltag-gstreamer.sh lines 123-126: taskset -c 4-7. Nothing checks PV_OPENCV_THREADS or that SCHED_FIFO was accepted (the native library only prints one line on stderr).
- First method from the review: Add 'int cpus' to Board (4 for both boards). At the start of runBoard, before prepareScene: apply PV_MLTAG_SIM_APP_CPUS if set; then assertEquals(board.cpus(), Runtime.getRuntime().availableProcessors()) with the message 'run with taskset -c a-d', unless PV_MLTAG_SIM_ALLOW_CPU_COUNT=1. Document 'taskset -c 0-3 ./gradlew ...' as the standard method (it also limits GC and JIT threads). Print in the mode line: PV_OPENCV_THREADS, PV_GSTREAMER_CAPTURE_RT_PRIORITY, VisionWorkerPool.THREADS, and at the end of each scenario the value of VisionWorkerPool.maxGroups(4) read from inside a loop thread. Proof: 'taskset -c 0-7' on an 8-CPU host must fail at once with the message; 'taskset -c 0-3' must run and print groups=1 for two cameras.
- Result of the independent check: real=True, confidence=high. Reasoning: I could not refute it. The fault is real but latent: the numbers measured on this host are not affected, because this host has 4 CPUs (/sys/devices/system/cpu/online = 0-3). I ran no build and no test; this is from reading the code only.

**What I confirmed**

- **No check of the CPU count.** `runBoard` (MLTagEndToEndSimulation.java:1840-1851) only prints "Computer: %d CPUs". `Board` has no CPU field. The example command in the class comment (lines 139-143) has no `taskset`, and line 128 expects runs on "a different computer". No test in photon-core/src/test refers to `VisionWorkerPool` or checks `availableProcessors`.
- **The CPU count changes the algorithm, not only the contention.** `makeSettings` sets `settings.threads = 4`. `AprilTagROIDecodePipe.process` (line 186) and `AprilTagPipeline.estimateSingleTagPoses` (line 636) call `VisionWorkerPool.maxGroups(4)`. Lines 82-90 of `VisionWorkerPool` give `min(4, THREADS + 1, max(1, (processors - cameras) / cameras))`.
  - 8 CPUs: 3 groups for two cameras, 4 for one camera.
  - 4 CPUs (both run scripts use `taskset -c 4-7`): 1 group for two cameras, 3 for one camera.
- **Concrete sequence.** Run the documented command on an 8-CPU computer. The two-camera scenario decodes tags and poses in 3 groups on 8 CPUs with no competition. The test passes, because it only asserts equality with the reference, and prints a normal report line with FPS values the board cannot reach.
- **`PV_MLTAG_SIM_APP_CPUS` is applied late.** `runScenario` lines 1504-1514 run after `prepareScene()` (line 1844). `prepareReference` runs a one-lane pipeline on the JUnit thread, so that thread loads `VisionWorkerPool` and creates the pool threads with the full mask. The mask is never restored, so later calibrations and the second `@Test` run under it while the first calibration did not.
- **Settings are not visible in the report.** `applyRealtimePriorityToCurrentThread` (GStreamerCaptureJNI.cpp:236-251) prints one stderr line when the kernel refuses SCHED_FIFO. Nothing prints `PV_OPENCV_THREADS` or the priority setting.

**Where the claim overstates**

- With `PV_MLTAG_SIM_APP_CPUS` set, the group count is already correct. On Linux HotSpot `Runtime.availableProcessors()` uses `sched_getaffinity(0)`, which is the mask of the calling thread (from my knowledge of HotSpot, not checked here). `maxGroups` reads it on each call from the masked pipeline thread.
- `THREADS` from the full count is then only an upper limit that does not bind. For the documented value `2,3` the group count is 1, so the unmasked pool threads are never used.
- What remains wrong with the late mask: JVM service threads (GC, JIT) keep all CPUs, and pool threads keep all CPUs whenever groups > 1.

**Faults in the proposed fix**

1. **Wrong order.** Applying the mask first and then `assertEquals(board.cpus(), availableProcessors())` compares 4 with 2 for the documented `PV_MLTAG_SIM_APP_CPUS=2,3`, so it always fails. It also fails for the second `@Test` in the same JVM (`forkEvery = 1` is per class, and the mask stays on the thread).
2. **The group print disturbs the measurement.** `maxGroups()` registers the calling thread as a camera for 1 second (`countActiveCallers`). With lanes = 2 the real callers are the MLTag-Frame-N threads, so a call from a SimVision thread doubles the camera count. For one camera on 4 CPUs it prints 1 where the pipelines used 3, and it lowers the real group count for 1 second.
3. **`taskset -c 0-3 ./gradlew` is not reliable.** gradle.properties sets `org.gradle.jvmargs`, so Gradle uses a daemon. A daemon that already runs keeps its own mask, and the test JVM is its child.
4. **Printing the priority variable does not show if SCHED_FIFO was accepted.**

Severity: low to medium. It is a latent validity gap, not a fault in the present numbers.
- Corrected method from the check (use THIS method unless the "Decisions" section says something different): All changes are in photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java. No production code changes.

**1. `Board` record**
- Add the field `int cpus`: the number of CPUs the run script gives the JVM. Both scripts use `taskset -c 4-7`, so `RUBIK` and `ORANGE_PI` get 4.
- Pass `defaultBoard.cpus()` in the `new Board(...)` copy in `runBoard`.

**2. New static state**
- `private static int hostCpus = -1;`
- `private static int appCpus;`
- `private static boolean cpuCountNotValid;`

**3. New method `private static synchronized void prepareCpus(Board board)`**

Call it in `runBoard` directly after the three `assumeTrue` lines and before `prepareScene()`.

a. One time only (`if (hostCpus < 0)`):
   - Read `hostCpus = Runtime.getRuntime().availableProcessors()` first, before any mask. On Linux this value is the CPU count of the calling thread's mask.
   - Set `appCpus = hostCpus`.
   - Move the `PV_MLTAG_SIM_APP_CPUS` block from `runScenario` (lines 1504-1514) to here and delete it there.
   - Assert that each CPU number is 0 to 63, build the mask, call `GStreamerCaptureJNI.setCurrentThreadCpuMask(mask)`, and set `appCpus = Long.bitCount(mask)`.
   - The one-time guard is necessary because the mask stays on the JUnit thread for the second `@Test`.
   - The reference run, the `VisionWorkerPool` class load, the pool threads and all calibrations then run under the mask, as with `taskset java`.

b. On each call, if `hostCpus != board.cpus()`:
   - If `PV_MLTAG_SIM_ALLOW_CPU_COUNT` is "1": set `cpuCountNotValid = true` and print a `line(...)`.
   - If not: `throw new AssertionError(...)` (or import `Assertions.fail`). The message must say:
     - "The simulation models a board with <board.cpus> CPUs, but this JVM can use <hostCpus>. The number of parallel groups of VisionWorkerPool and the competition for the CPUs are then different from the board."
     - "Run: `./gradlew --stop`; then `taskset -c <4 CPUs on 4 different physical cores, see lscpu -e> ./gradlew --no-daemon :photon-core:test --tests ...`. A Gradle daemon that already runs keeps its own CPU mask."
     - "Set PV_MLTAG_SIM_ALLOW_CPU_COUNT=1 to run anyway."

**4. Mode line in `runBoard`**

Keep it after `prepareScene()`. Print:
- `hostCpus` (not a new `availableProcessors()` call) and `appCpus`
- `VisionWorkerPool.THREADS`
- `PV_OPENCV_THREADS` and `Core.getNumThreads()`
- `PV_GSTREAMER_CAPTURE_RT_PRIORITY`, `PV_GSTREAMER_CAPTURE_DECODE_CPUS`, `PV_MLTAG_SIM_APP_CPUS`

**5. Group count in `report()`**
- Do not call `VisionWorkerPool.maxGroups()` from the JUnit thread or a SimVision thread.
- Print the expected value from the same formula as `VisionWorkerPool.maxGroups` lines 82-90, with a comment that names those lines. It holds for the default `PV_MLTAG_ROI_POOL_ADAPTIVE=true`:
  `groups = max(1, min(4, min(VisionWorkerPool.MAX_GROUPS, VisionWorkerPool.THREADS + 1), max(1, (appCpus - cameras) / cameras)))`
- Add `groups=%d` to the report line.
- If `cpuCountNotValid`, append " | NOT VALID FOR THE BOARD: <hostCpus> CPUs" to each report line.

**6. Real-time priority check in `runScenario`**

After the 5 s warm-up:
- Count the threads with SCHED_FIFO. For each `/proc/self/task/*/stat`, take the text after the last ')', split on spaces, and test index 38 (field 41, policy) for "1".
- Pass the count to the report and print it as `rtThreads=%d`.
- If `PV_GSTREAMER_CAPTURE_RT_PRIORITY` parses to more than 0 and the count is less than 2 x cameras (camera source thread and JPEG decode thread for each camera), fail with: "The kernel refused SCHED_FIFO (needs root or CAP_SYS_NICE). Remove PV_GSTREAMER_CAPTURE_RT_PRIORITY or give the permission."

**7. Class comment**
- State that the test JVM must have exactly the CPUs of the board.
- Put the `./gradlew --stop` and `taskset ... --no-daemon` method in the example command.
- State that `PV_MLTAG_SIM_APP_CPUS` does not move the GC and JIT threads, and that `taskset` on the JVM does.

**Proof**

- **Wrong CPU count, on this 4-CPU host:** `./gradlew --stop; PV_MLTAG_SIM=1 taskset -c 0-2 ./gradlew --no-daemon :photon-core:test --tests "*MLTagEndToEndSimulation.orangePi5"` must fail in seconds, before the "frame set:" lines, with "this JVM can use 3". The same holds on an 8-CPU computer with no `taskset`.
- **Override:** the same command with `PV_MLTAG_SIM_ALLOW_CPU_COUNT=1` must run, and each report line must carry the NOT VALID mark.
- **Normal run on 4 CPUs:** it must run and print `groups=3` for cameras=1 and `groups=1` for cameras=2. The reviewer's method would print 1 for cameras=1 on the Orange Pi profile, so this also proves point 5.
- **Mask:** `PV_MLTAG_SIM_APP_CPUS=2,3 PV_GSTREAMER_CAPTURE_DECODE_CPUS=0,1` on 4 CPUs must pass the check and print `appCpus=2`, pool threads = 1 and `groups=1`. The reviewer's order would fail here with 4 != 2. During the run, `grep Cpus_allowed_list /proc/<test JVM pid>/task/*/status` must show 2-3 for the SimVision, MLTag-Frame and AprilTag-ROI-Decode threads.
- **Both tests in one JVM:** `--tests "*MLTagEndToEndSimulation"` with `PV_MLTAG_SIM_APP_CPUS` set must pass the check in the second test too.
- **Priority:** with `PV_GSTREAMER_CAPTURE_RT_PRIORITY=10` and no CAP_SYS_NICE the run must fail with the SCHED_FIFO message. As root it must print `rtThreads` >= 2 x cameras.

### G6: compareWithReference accepts NaN, and an empty fallback result passes on each frame that has no tag; alternate pose, ambiguity and multi-tag result are not compared
- Location: /home/user/photonvision-mltag/photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java:917
- Failure scenario: (1) A load result with a NaN corner or NaN pose value: Math.abs(expected - NaN) > 1e-9 is false, so the value counts as equal. (2) finishFrame or awaitResult catch an exception and return a result with no targets; for a frame whose reference list is empty (regions without a tag, sources where the detector found no tag) this is 'equal', so a pipeline that fails on such frames shows 100%. (3) With PV_MLTAG_SIM_MULTITAG=1 a wrong multiTagResult or a wrong alternate pose is not seen. (4) With PV_MLTAG_SIM_NULL=1 the loop does resultsEqualToReference++ with no comparison (line 1409) and the report prints equalToReference=100.00%.
- Evidence: Line 917: if (Math.abs(expected.get(tag)[i] - tags.get(tag)[i]) > 1e-9). Line 911: only the list size is compared for empty lists. tagData (877-900) has no alternate pose, ambiguity or multi-tag data. AprilTagPipeline.finishFrame (lines 414-422) and awaitResult (437) return empty results after an exception.
- First method from the review: In compareWithReference use: double a = expected, b = actual; if (Double.doubleToLongBits(a) != Double.doubleToLongBits(b) && !(Math.abs(a - b) <= 1e-9)) -> difference. Add to tagData: alternate pose (6 values), pose ambiguity, and for the result the multi-tag best pose and the fiducial ID list (compare when present in the reference). Count frames with an empty reference separately and print 'resultsWithTags=' so the reader sees how many comparisons had data. For the empty fallback: add a counter in the test for results where processingNanos == 0 and targets is empty while the model gave regions, and assert 0. For the null pipeline print 'equalToReference=n/a'. Proof: unit test with NaN in the actual list must return a text; a SimulatedModel that throws on each 10th frame must make the run fail.
- Note of the lead engineer: Do only the NaN part of this item.

### G5: runScenario does not restore the test state on a failure: the extra JPEG decode stays on for later tests in the same JVM, and captures and pipelines are not released
- Location: /home/user/photonvision-mltag/photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java:1493
- Failure scenario: PV_MLTAG_SIM_MODE=work with k=2.0: setTestExtraDecodePercent(100). A loop fails ('The vision loop failed', line 1618) or the run gives no result (line 1709). The exception leaves runScenario before line 1711, so testExtraDecodePercent stays 100 in the process. Tests that run later in the same Gradle test JVM (MLTagStageBenchmark.jpegDecodeStage, the timing tests of GStreamerCaptureTest) then decode each JPEG two times: decode times are 2 x and timing asserts can fail for a wrong reason. In the same exit path the captures and pipelines of the loops after the failed one are not released, and after thread.join(5000) times out the code releases a pipeline and a capture that a loop thread can still use.
- Evidence: Lines 1493-1494 set the value, line 1711 resets it, with no try/finally. Lines 1612-1639: the throw at 1618 is inside the loop that releases pipeline and camera. GStreamerCaptureJNI.cpp line 78: the value is one process-wide atomic.
- First method from the review: Put the body of runScenario after the setTestExtraDecodePercent call in try { ... } finally { GStreamerCaptureJNI.setTestExtraDecodePercent(0); stop all loops; join each thread; if a thread is still alive after the join, do not release its pipeline and camera and fail with 'loop did not stop'; else release each pipeline and camera }. Collect the first loop failure and throw it after the cleanup. Proof: a test that sets PV_MLTAG_SIM_FAULT=throw (SimulatedModel.detect throws after 100 frames) must end with the assertion error and a later capture in the same JVM must show the same convertNanos for each frame as before the scenario.

