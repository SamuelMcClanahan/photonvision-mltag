# Specification H: capture time from the native monotonic clock

Files that you can edit:
- photon-core/src/main/java/org/photonvision/gstreamer/GStreamerFrame.java
- photon-core/src/main/java/org/photonvision/vision/frame/provider/GStreamerUSBFrameProvider.java
- tools/jetson-gstreamer-capture-jni/src/GStreamerCaptureJNI.cpp
- photon-core/src/test/java/org/photonvision/vision/frame/provider/MLTagEndToEndSimulation.java (only SimulatedCamera.getInputMat)
- photon-core/src/test/java/org/photonvision/gstreamer/GStreamerFrameTest.java (new)

## Problem (found by the check of specification G, item G3)
GStreamerFrame.toCaptureMetadata computes the capture time as a Java clock value (MathUtils.wpiNanoTime(), read by the caller after grab() returns) minus estimatedAgeNanos (computed by native code before the return). The time between the native age computation and the Java clock read is not in the age. A JVM safepoint pause (G1 young collection, 12 to 15 ms measured) at the return of grab() makes the capture time of that frame too new by the remaining pause (11.9 ms measured with -Xlog:safepoint). The robot code uses this time with the pose, so the pose gets a time that is too new, and two results can have capture times out of order although the frames are in order. MLTagEndToEndSimulation requirement 2 (capture time of a result > capture time of the previous result) failed in 4 of 4 time scaling runs for this cause. The computation is in the code since commit a0ebab6.

## Decisions of the lead engineer
- H1 (native): in frameToResult of GStreamerCaptureJNI.cpp read the monotonic clock one time (monoNanos()) and use this value for the estimated age and for the pull time stamp (field 9). Now the age uses monoNanos() and the pull time stamp a later steadyNowNanos(), so pull - age is not exactly the native capture time.
- H2 (Java): GStreamerFrame.toCaptureMetadata(String source) has no clock argument. It computes the native monotonic capture time (nativePullTimestampNanos - estimatedAgeNanos; nativePullTimestampNanos if the age is not known) and converts it to the WPILib time with an offset from two Java clock reads directly one after the other: System.nanoTime(), then MathUtils.wpiNanoTime(). System.nanoTime() on Linux is CLOCK_MONOTONIC, the clock of monoNanos(). A pause between the two reads moves the offset, so read System.nanoTime() again after wpiNanoTime(); if the two System.nanoTime() reads are more than 1 ms apart, read again (at most 5 times) and keep the pair with the smallest gap. Use the middle of the two System.nanoTime() reads.
- H3: GStreamerUSBFrameProvider.grabGStreamerFrame and SimulatedCamera.getInputMat of MLTagEndToEndSimulation both use this function.
- Test (no hardware, no switch): GStreamerFrameTest: a frame with a native capture time 40 ms ago gives a capture time that is 40 ms before the WPILib time (within 2 ms), and two calls 30 ms apart give the same capture time (within 1 ms). With the old computation (clock read at the call minus the age) the second call is 30 ms newer.
- Check: the two commands of specification G (work mode and time scaling) must each run 5 or more times, each scenario line with equalToReference=100.00% and orderAndTimeErrors=0. Requirement 2 of orderAndTimeError must not be made weaker.

## Rules for the implementer
The rules of specification G apply (work tree, build and test commands, both native builds must compile, no commit, no push, no spotless).
