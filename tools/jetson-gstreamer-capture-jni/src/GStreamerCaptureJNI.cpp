#include <jni.h>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <opencv2/core.hpp>

#include <pthread.h>
#include <sched.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <csetjmp>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#ifdef PV_HAVE_LIBJPEG
extern "C" {
#include <jpeglib.h>

#include <jerror.h>
}
#endif

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#define PV_HAVE_NEON 1
#endif

namespace {

constexpr int kFieldCount = 14;

// Values of the outputMode argument of createWithOptions(). Only compressed (image/jpeg) input
// uses this value. Raw input keeps the pixel layout that the pipeline gives.
constexpr int kOutputAuto = 0;  // JPEG -> BGR (1-component JPEG -> GRAY)
constexpr int kOutputGray = 1;  // JPEG -> GRAY (luma only, chroma is not decoded)
constexpr int kOutputBgr = 2;  // JPEG -> BGR

// Bits of getCapabilities().
constexpr jlong kCapOptions = 1;
constexpr jlong kCapJpegDecode = 2;
constexpr jlong kCapStats = 4;
constexpr jlong kCapLumaFormats = 8;
constexpr jlong kCapThreadAffinity = 16;
constexpr jlong kCapJpegIdct = 32;

constexpr int kStatsFieldCount = 16;

// When the worker for compressed input starts the decode of the next frame. Set it with
// PV_GSTREAMER_CAPTURE_DECODE_POLICY (newest, demand, jit).
//   Newest (default): at once. The newest decoded frame replaces an older one that was not
//     taken. This gives the lowest frame age.
//   Demand: after the consumer took the last decoded frame.
//   JustInTime: as Demand, and if the consumer is slower than the camera, the start waits so
//     that the decode ends when the consumer is expected to come back. No frame is decoded that
//     the consumer does not use, which decreases the processor load and the heat. In a
//     simulation with two cameras the frame rate was the same as with Newest and the latency
//     was some milliseconds higher.
enum class DecodePolicy { Newest, Demand, JustInTime };

// Test aid for simulations of a slower processor: after each JPEG frame, decode this percent of
// the frame again and discard the pixels. 0 (default) = no extra work. Only
// setTestExtraDecodePercent() changes it.
std::atomic<int> testExtraDecodePercent {0};

std::once_flag gstInitFlag;
std::mutex gstCreateMutex;

void ensureGstInit() {
  std::call_once(gstInitFlag, []() {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
  });
}

template <typename T>
jlong toHandle(T* ptr) {
  return static_cast<jlong>(reinterpret_cast<intptr_t>(ptr));
}

template <typename T>
T* fromHandle(jlong handle) {
  return reinterpret_cast<T*>(static_cast<intptr_t>(handle));
}

std::string toString(JNIEnv* env, jstring value) {
  if (value == nullptr) {
    return {};
  }

  const char* chars = env->GetStringUTFChars(value, nullptr);
  if (chars == nullptr) {
    return {};
  }

  std::string result(chars);
  env->ReleaseStringUTFChars(value, chars);
  return result;
}

std::string envLower(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr) {
    return {};
  }
  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

bool envFlagEnabled(const char* name) {
  const std::string value = envLower(name);
  if (value.empty()) {
    return false;
  }
  return value != "0" && value != "false" && value != "no" && value != "off";
}

int envInt(const char* name, int defaultValue) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return defaultValue;
  }

  char* end = nullptr;
  const long parsed = std::strtol(raw, &end, 10);
  if (end == raw || parsed < 0 || parsed > 1000000L) {
    return defaultValue;
  }
  return static_cast<int>(parsed);
}

jlongArray emptyResult(JNIEnv* env) {
  return env->NewLongArray(0);
}

jlong gstTimeToJlong(GstClockTime time) {
  return GST_CLOCK_TIME_IS_VALID(time) ? static_cast<jlong>(time) : static_cast<jlong>(-1);
}

// CLOCK_MONOTONIC in nanoseconds. The source pad probe and grab() both use this clock, so a frame
// age does not depend on buffer timestamps that downstream elements can rewrite.
int64_t monoNanos() {
  struct timespec ts {};
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return static_cast<int64_t>(ts.tv_sec) * 1000000000LL + static_cast<int64_t>(ts.tv_nsec);
}

// Limits the calling thread to the CPUs of a bit mask (bit n = CPU n). Returns true if the
// kernel accepted the mask.
bool setCurrentThreadCpuMask(uint64_t mask) {
  cpu_set_t set;
  CPU_ZERO(&set);
  for (int cpu = 0; cpu < 64; ++cpu) {
    if ((mask >> cpu) & 1ULL) {
      CPU_SET(cpu, &set);
    }
  }
  const pid_t tid = static_cast<pid_t>(syscall(SYS_gettid));
  return sched_setaffinity(tid, sizeof(set), &set) == 0;
}

// PV_GSTREAMER_CAPTURE_DECODE_CPUS is a list of CPU numbers, for example "4,5". The decode thread
// of the first camera uses the first CPU of the list, the second camera the second CPU, and so
// on. With the other threads of the application on other CPUs (taskset), the time when a camera
// frame is available does not depend on the load of the vision pipelines.
std::atomic<int> decodeWorkerCount {0};

[[maybe_unused]] void pinDecodeWorkerFromEnvironment() {
  const char* raw = std::getenv("PV_GSTREAMER_CAPTURE_DECODE_CPUS");
  if (raw == nullptr || raw[0] == '\0') {
    return;
  }

  std::vector<int> cpus;
  const char* cursor = raw;
  while (*cursor != '\0') {
    char* end = nullptr;
    const long value = std::strtol(cursor, &end, 10);
    if (end == cursor) {
      ++cursor;
      continue;
    }
    if (value >= 0 && value < 64) {
      cpus.push_back(static_cast<int>(value));
    }
    cursor = end;
  }
  if (cpus.empty()) {
    return;
  }

  const int index = decodeWorkerCount.fetch_add(1);
  const int cpu = cpus[static_cast<size_t>(index) % cpus.size()];
  if (!setCurrentThreadCpuMask(1ULL << cpu)) {
    std::cerr << "[GStreamer Capture JNI] Could not put the decode thread on CPU " << cpu << "\n";
  }
}

// PV_GSTREAMER_CAPTURE_RT_PRIORITY=N (1 to 50) gives the capture threads the real-time policy
// SCHED_FIFO with priority N: the GStreamer thread that takes the frames from the camera, and the
// JPEG decode thread. These threads then get a CPU immediately when a frame arrives, also when the
// vision pipelines use all CPUs. So the time when a frame is available does not depend on the
// pipeline load. The threads do a limited amount of work for each frame and then wait, so they
// cannot stop the other threads. The process needs the permission for real-time priority (root or
// CAP_SYS_NICE). If the kernel refuses, the threads keep the normal policy.
int realtimePriorityFromEnvironment() {
  static const int priority = [] {
    const char* raw = std::getenv("PV_GSTREAMER_CAPTURE_RT_PRIORITY");
    if (raw == nullptr || raw[0] == '\0') {
      return 0;
    }
    const long value = std::strtol(raw, nullptr, 10);
    return static_cast<int>(std::max(0L, std::min(50L, value)));
  }();
  return priority;
}

// GStreamer streaming threads come from a shared thread pool. After a capture stops, a thread can
// run a task of another pipeline (for example the legacy software decode). So the policy before
// the change is kept here, and the bus sync handler restores it when the task leaves the thread.
thread_local bool realtimePromoted = false;
thread_local int realtimeSavedPolicy = SCHED_OTHER;
thread_local sched_param realtimeSavedParam {};
// True after the source probe tried to set the priority on this thread.
thread_local bool sourceThreadPriorityApplied = false;

void applyRealtimePriorityToCurrentThread(const char* threadName) {
  const int priority = realtimePriorityFromEnvironment();
  if (priority <= 0) {
    return;
  }
  int savedPolicy = SCHED_OTHER;
  sched_param savedParam {};
  const bool saved = pthread_getschedparam(pthread_self(), &savedPolicy, &savedParam) == 0;
  sched_param param {};
  param.sched_priority = priority;
  const int error = pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
  if (error == 0 && saved && !realtimePromoted) {
    realtimePromoted = true;
    realtimeSavedPolicy = savedPolicy;
    realtimeSavedParam = savedParam;
  }
  if (error != 0) {
    static std::atomic<bool> reported {false};
    if (!reported.exchange(true)) {
      std::cerr << "[GStreamer Capture JNI] Could not set real-time priority " << priority
                << " for the " << threadName << " thread: " << std::strerror(error) << "\n";
    }
  }
}

jlong steadyNowNanos() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<jlong>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

// ---------------------------------------------------------------------------------------------
// Pixel buffer pool. Decoded frames are large (2 MP and more). Reuse of the buffers prevents a
// page-fault storm on each frame.
// ---------------------------------------------------------------------------------------------

struct PooledBuffer {
  uint8_t* data = nullptr;
  size_t capacity = 0;
};

class BufferPool {
 public:
  ~BufferPool() {
    for (PooledBuffer& buffer : free_) {
      std::free(buffer.data);
    }
  }

  PooledBuffer acquire(size_t size) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      for (size_t i = 0; i < free_.size(); ++i) {
        if (free_[i].capacity >= size) {
          PooledBuffer buffer = free_[i];
          free_.erase(free_.begin() + static_cast<std::ptrdiff_t>(i));
          return buffer;
        }
      }
    }

    PooledBuffer buffer;
    // Round up to a multiple of the alignment, plus padding for SIMD loops that read ahead.
    const size_t rounded = ((size + 63) / 64) * 64 + 64;
    buffer.data = static_cast<uint8_t*>(std::aligned_alloc(64, rounded));
    buffer.capacity = buffer.data != nullptr ? rounded : 0;
    return buffer;
  }

  void release(PooledBuffer buffer) {
    if (buffer.data == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(mutex_);
    if (free_.size() >= kMaxIdleBuffers) {
      std::free(buffer.data);
      return;
    }
    free_.push_back(buffer);
  }

 private:
  static constexpr size_t kMaxIdleBuffers = 4;
  std::mutex mutex_;
  std::vector<PooledBuffer> free_;
};

// ---------------------------------------------------------------------------------------------
// Frame holder
// ---------------------------------------------------------------------------------------------

struct FrameHolder {
  // ABI NOTE: tools/jetson-trt-jni/src/TensorRTJNI.cpp has a copy of the fields from `sample` to
  // `mlHeight` (GStreamerFrameHolder). Do not change the order or the types of these fields. Add
  // new fields only after `mlHeight`.
  GstSample* sample = nullptr;
  GstBuffer* buffer = nullptr;
  GstSample* mlSample = nullptr;
  GstBuffer* mlBuffer = nullptr;
  GstMapInfo map {};
  bool mapped = false;
  cv::Mat* mat = nullptr;
  int sourceWidth = 0;
  int sourceHeight = 0;
  int mlWidth = 0;
  int mlHeight = 0;

  // Fields that only this library uses.
  GstVideoFrame videoFrame {};
  bool videoFrameMapped = false;
  std::shared_ptr<BufferPool> pool;
  PooledBuffer pooled {};
  int channels = 0;
  int64_t captureMonoNanos = -1;
  jlong ptsNanos = -1;
  jlong dtsNanos = -1;
  jlong durationNanos = -1;
  jlong ptsFallbackAgeNanos = -1;
};

void releaseFrameHolder(FrameHolder* holder) {
  if (holder == nullptr) {
    return;
  }

  // Java's OpenCV Mat wrapper owns this cv::Mat header after new Mat(nativeAddr). Deleting it here
  // races/double-frees with Mat.finalize(); this holder only owns the pixel memory lifetime.
  holder->mat = nullptr;

  if (holder->videoFrameMapped) {
    gst_video_frame_unmap(&holder->videoFrame);
    holder->videoFrameMapped = false;
  }

  if (holder->mapped && holder->buffer != nullptr) {
    gst_buffer_unmap(holder->buffer, &holder->map);
    holder->mapped = false;
  }

  if (holder->sample != nullptr) {
    gst_sample_unref(holder->sample);
    holder->sample = nullptr;
  }
  if (holder->mlSample != nullptr) {
    gst_sample_unref(holder->mlSample);
    holder->mlSample = nullptr;
    holder->mlBuffer = nullptr;
  }

  if (holder->pool != nullptr) {
    holder->pool->release(holder->pooled);
    holder->pooled = {};
    holder->pool.reset();
  }

  delete holder;
}

// Deletes a holder that was never given to Java. In that case the cv::Mat header is still ours.
void discardFrameHolder(FrameHolder* holder) {
  if (holder == nullptr) {
    return;
  }
  delete holder->mat;
  holder->mat = nullptr;
  releaseFrameHolder(holder);
}

// ---------------------------------------------------------------------------------------------
// Capture state
// ---------------------------------------------------------------------------------------------

struct CaptureStats {
  std::atomic<int64_t> sourceBuffers {0};
  std::atomic<int64_t> samplesPulled {0};
  std::atomic<int64_t> framesProduced {0};
  std::atomic<int64_t> framesDelivered {0};
  std::atomic<int64_t> framesReplaced {0};
  std::atomic<int64_t> decodeErrors {0};
  std::atomic<int64_t> corruptFrames {0};
  std::atomic<int64_t> convertNanosSum {0};
  std::atomic<int64_t> convertNanosMax {0};
  std::atomic<int64_t> ageSamples {0};
  std::atomic<int64_t> ageNanosSum {0};
  std::atomic<int64_t> ageNanosMax {0};
  std::atomic<int64_t> deviceDelayNanosSum {0};
  std::atomic<int64_t> sourcePtsInvalid {0};
  std::atomic<int64_t> captureTimeMissing {0};
  std::atomic<int64_t> ptsAheadOfClock {0};
};

void atomicMax(std::atomic<int64_t>& target, int64_t value) {
  int64_t previous = target.load(std::memory_order_relaxed);
  while (value > previous &&
         !target.compare_exchange_weak(previous, value, std::memory_order_relaxed)) {
  }
}

struct Capture {
  GstElement* pipeline = nullptr;
  GstAppSink* appsink = nullptr;
  GstAppSink* mlAppsink = nullptr;
  GstSample* pendingMlSample = nullptr;
  GstClock* clock = nullptr;
  GstClockTime baseTime = GST_CLOCK_TIME_NONE;
  GstCaps* captureTimeCaps = nullptr;
  GstPad* sourcePad = nullptr;
  gulong sourceProbeId = 0;
  uint64_t sequence = 0;
  int outputMode = kOutputAuto;
  bool wrapMappedBuffer = false;
  bool strictMlPts = false;
  DecodePolicy decodePolicy = DecodePolicy::Newest;
  bool tolerateExtraneousJpegBytes = false;
  // The inverse DCT of the JPEG decode. True: the accurate integer method (JDCT_ISLOW). False:
  // the fast method (JDCT_IFAST), which is the default of the GStreamer jpegdec element. It is
  // atomic because a test can change it while the worker decodes.
  std::atomic<bool> jpegIdctAccurate {true};
  int strictMlMaxPulls = 4;
  int strictCpuMaxDrops = 2;
  int timingLogEvery = 0;
  int timingFrames = 0;
  int timingMlMatched = 0;
  int timingMlMissing = 0;
  int timingStrictCpuDrops = 0;
  int timingStrictMlDrops = 0;
  double timingMlPtsDeltaMsSum = 0.0;
  double timingMlPtsDeltaMsMax = 0.0;

  std::shared_ptr<BufferPool> pool = std::make_shared<BufferPool>();
  CaptureStats stats;
  int64_t statsLogLastNanos = 0;
  int64_t statsLogLast[kStatsFieldCount] = {};

  // Compressed-input worker. It holds at most one decoded frame (the newest one).
  std::thread worker;
  std::atomic<bool> workerRunning {false};
  std::atomic<bool> workerStop {false};
  std::mutex slotMutex;
  std::condition_variable slotFilled;
  std::condition_variable slotEmptied;
  FrameHolder* latest = nullptr;
  bool workerEnded = false;

  // Timing data for the JustInTime policy. slotMutex guards these fields.
  bool consumerWaiting = false;
  bool lastTakeWaited = true;
  int64_t lastTakeNanos = 0;
  int64_t takeIntervalNanos[4] = {};
  int takeIntervalCount = 0;
  int takeIntervalNext = 0;
  std::atomic<int64_t> decodeNanosAverage {0};
  std::atomic<int64_t> sourcePeriodNanosAverage {0};
  int64_t lastSourceNanos = 0;
  // Window counts for suspect JPEG frames (see frameFromJpegSample). Only the thread that
  // decodes writes these fields.
  int jpegWindowFrames = 0;
  int jpegWindowSuspect = 0;
  bool unsupportedFormatLogged = false;
};

void stopPipeline(GstElement* pipeline) {
  if (pipeline == nullptr) {
    return;
  }

  GstStateChangeReturn stateResult = gst_element_set_state(pipeline, GST_STATE_NULL);
  if (stateResult == GST_STATE_CHANGE_FAILURE) {
    std::cerr << "[GStreamer Capture JNI] Failed to set pipeline NULL during cleanup\n";
    return;
  }

  GstState state = GST_STATE_VOID_PENDING;
  GstState pending = GST_STATE_VOID_PENDING;
  stateResult = gst_element_get_state(pipeline, &state, &pending, 2 * GST_SECOND);
  if (stateResult == GST_STATE_CHANGE_FAILURE) {
    std::cerr << "[GStreamer Capture JNI] Pipeline failed while entering NULL during cleanup\n";
  }
}

void stopWorker(Capture* capture) {
  if (capture == nullptr) {
    return;
  }
  capture->workerStop.store(true);
  {
    std::lock_guard<std::mutex> lock(capture->slotMutex);
    capture->slotEmptied.notify_all();
    capture->slotFilled.notify_all();
  }
  if (capture->worker.joinable()) {
    capture->worker.join();
  }
  capture->workerRunning.store(false);

  std::lock_guard<std::mutex> lock(capture->slotMutex);
  if (capture->latest != nullptr) {
    discardFrameHolder(capture->latest);
    capture->latest = nullptr;
  }
}

void destroyCapture(Capture* capture) {
  if (capture == nullptr) {
    return;
  }

  stopWorker(capture);

  if (capture->sourcePad != nullptr) {
    if (capture->sourceProbeId != 0) {
      gst_pad_remove_probe(capture->sourcePad, capture->sourceProbeId);
      capture->sourceProbeId = 0;
    }
    gst_object_unref(capture->sourcePad);
    capture->sourcePad = nullptr;
  }

  if (capture->pipeline != nullptr) {
    stopPipeline(capture->pipeline);
  }

  if (capture->appsink != nullptr) {
    gst_object_unref(capture->appsink);
    capture->appsink = nullptr;
  }
  if (capture->mlAppsink != nullptr) {
    gst_object_unref(capture->mlAppsink);
    capture->mlAppsink = nullptr;
  }
  if (capture->pendingMlSample != nullptr) {
    gst_sample_unref(capture->pendingMlSample);
    capture->pendingMlSample = nullptr;
  }

  if (capture->clock != nullptr) {
    gst_object_unref(capture->clock);
    capture->clock = nullptr;
  }

  if (capture->captureTimeCaps != nullptr) {
    gst_caps_unref(capture->captureTimeCaps);
    capture->captureTimeCaps = nullptr;
  }

  if (capture->pipeline != nullptr) {
    gst_object_unref(capture->pipeline);
    capture->pipeline = nullptr;
  }

  delete capture;
}

// ---------------------------------------------------------------------------------------------
// Capture time
// ---------------------------------------------------------------------------------------------

// Runs on the streaming thread of the source element for each buffer. It records the capture time
// on the monotonic clock in a GstReferenceTimestampMeta. Decoders, parsers and converters copy this
// meta, but they can rewrite PTS (a parser that counts frames makes PTS drift from the real time).
GstPadProbeReturn sourceBufferProbe(GstPad*, GstPadProbeInfo* info, gpointer userData) {
  auto* capture = static_cast<Capture*>(userData);
  GstBuffer* buffer = GST_PAD_PROBE_INFO_BUFFER(info);
  if (capture == nullptr || buffer == nullptr) {
    return GST_PAD_PROBE_OK;
  }

  // This function runs on the GStreamer thread that takes the frames from the camera.
  if (!sourceThreadPriorityApplied) {
    sourceThreadPriorityApplied = true;
    applyRealtimePriorityToCurrentThread("camera source");
  }

  const int64_t now = monoNanos();
  int64_t deviceDelay = 0;

  // At this point PTS is the value from the source element. For v4l2src it is the kernel time of
  // the start of the frame, in pipeline running time. The difference to "now" is the time that
  // the frame used to arrive (USB transfer and driver queue).
  const GstClockTime pts = GST_BUFFER_PTS(buffer);
  bool ptsUsable = false;
  if (GST_CLOCK_TIME_IS_VALID(pts) && capture->pipeline != nullptr) {
    GstClock* clock = GST_ELEMENT_CLOCK(capture->pipeline);
    const GstClockTime baseTime = gst_element_get_base_time(capture->pipeline);
    if (clock != nullptr && GST_CLOCK_TIME_IS_VALID(baseTime)) {
      const GstClockTime clockNow = gst_clock_get_time(clock);
      if (GST_CLOCK_TIME_IS_VALID(clockNow) && clockNow >= baseTime + pts) {
        const GstClockTime delay = clockNow - (baseTime + pts);
        if (delay <= 500 * GST_MSECOND) {
          deviceDelay = static_cast<int64_t>(delay);
          ptsUsable = true;
        }
      }
    }
  }
  if (!ptsUsable) {
    capture->stats.sourcePtsInvalid.fetch_add(1, std::memory_order_relaxed);
  }

  if (!gst_buffer_is_writable(buffer)) {
    buffer = gst_buffer_make_writable(buffer);
    GST_PAD_PROBE_INFO_DATA(info) = buffer;
  }
  gst_buffer_add_reference_timestamp_meta(
      buffer, capture->captureTimeCaps, static_cast<GstClockTime>(now - deviceDelay),
      GST_CLOCK_TIME_NONE);

  // Camera frame period, for the JustInTime decode policy. Only this thread writes these fields.
  if (capture->lastSourceNanos > 0) {
    const int64_t period = now - capture->lastSourceNanos;
    if (period > 0 && period < 1000000000LL) {
      const int64_t previous = capture->sourcePeriodNanosAverage.load(std::memory_order_relaxed);
      capture->sourcePeriodNanosAverage.store(
          previous == 0 ? period : (previous * 7 + period) / 8, std::memory_order_relaxed);
    }
  }
  capture->lastSourceNanos = now;

  capture->stats.sourceBuffers.fetch_add(1, std::memory_order_relaxed);
  capture->stats.deviceDelayNanosSum.fetch_add(deviceDelay, std::memory_order_relaxed);
  return GST_PAD_PROBE_OK;
}

// Installs the capture-time probe. The probe is on the src pad of the element with the name
// "pvcapturetime", if the pipeline has one. If not, it is on the src pad of the first source
// element (the camera). The name lets a pipeline define the point where a frame "arrives".
void installSourceProbe(Capture* capture) {
  GstPad* pad = nullptr;

  GstElement* named = gst_bin_get_by_name(GST_BIN(capture->pipeline), "pvcapturetime");
  if (named != nullptr) {
    pad = gst_element_get_static_pad(named, "src");
    gst_object_unref(named);
  }

  if (pad == nullptr) {
    GstIterator* sources = gst_bin_iterate_sources(GST_BIN(capture->pipeline));
    if (sources != nullptr) {
      GValue item = G_VALUE_INIT;
      if (gst_iterator_next(sources, &item) == GST_ITERATOR_OK) {
        GstElement* source = GST_ELEMENT(g_value_get_object(&item));
        pad = source != nullptr ? gst_element_get_static_pad(source, "src") : nullptr;
        g_value_reset(&item);
      }
      g_value_unset(&item);
      gst_iterator_free(sources);
    }
  }

  if (pad != nullptr) {
    capture->sourcePad = pad;
    capture->sourceProbeId =
        gst_pad_add_probe(pad, GST_PAD_PROBE_TYPE_BUFFER, sourceBufferProbe, capture, nullptr);
  }
}

int64_t captureTimeFromBuffer(Capture* capture, GstBuffer* buffer) {
  if (capture == nullptr || buffer == nullptr || capture->captureTimeCaps == nullptr) {
    return -1;
  }
  GstReferenceTimestampMeta* meta =
      gst_buffer_get_reference_timestamp_meta(buffer, capture->captureTimeCaps);
  if (meta == nullptr || !GST_CLOCK_TIME_IS_VALID(meta->timestamp)) {
    return -1;
  }
  return static_cast<int64_t>(meta->timestamp);
}

// Old estimate from the buffer PTS and the pipeline clock. It is only a fallback for buffers that
// lost the capture-time meta. It returns -1 (unknown) when PTS is ahead of the clock, because a
// PTS in the future is proof that an element rewrote it.
jlong ptsEstimatedAgeNanos(Capture* capture, GstBuffer* buffer) {
  if (capture == nullptr || buffer == nullptr) {
    return -1;
  }

  const GstClockTime pts = GST_BUFFER_PTS(buffer);
  if (!GST_CLOCK_TIME_IS_VALID(pts)) {
    return -1;
  }

  if (capture->clock == nullptr) {
    capture->clock = gst_element_get_clock(capture->pipeline);
  }
  capture->baseTime = gst_element_get_base_time(capture->pipeline);

  if (capture->clock == nullptr || !GST_CLOCK_TIME_IS_VALID(capture->baseTime)) {
    return -1;
  }

  const GstClockTime now = gst_clock_get_time(capture->clock);
  const GstClockTime absolutePts = capture->baseTime + pts;
  if (!GST_CLOCK_TIME_IS_VALID(now)) {
    return -1;
  }
  if (now < absolutePts) {
    capture->stats.ptsAheadOfClock.fetch_add(1, std::memory_order_relaxed);
    return -1;
  }

  return static_cast<jlong>(now - absolutePts);
}

// Runs on the thread that posts the message. GStreamer posts STREAM_STATUS LEAVE from the
// streaming thread itself when its task ends, so the restore below acts on the correct thread.
// It passes each message on, so logBusFailure still sees errors.
GstBusSyncReply restorePriorityBusHandler(GstBus*, GstMessage* message, gpointer) {
  if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_STREAM_STATUS) {
    GstStreamStatusType type = GST_STREAM_STATUS_TYPE_CREATE;
    GstElement* owner = nullptr;
    gst_message_parse_stream_status(message, &type, &owner);
    if (type == GST_STREAM_STATUS_TYPE_LEAVE && realtimePromoted) {
      pthread_setschedparam(pthread_self(), realtimeSavedPolicy, &realtimeSavedParam);
      realtimePromoted = false;
      sourceThreadPriorityApplied = false;
    }
  }
  return GST_BUS_PASS;
}

bool logBusFailure(Capture* capture) {
  if (capture == nullptr || capture->pipeline == nullptr) {
    return false;
  }

  GstBus* bus = gst_element_get_bus(capture->pipeline);
  if (bus == nullptr) {
    return false;
  }

  GstMessage* message =
      gst_bus_pop_filtered(bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  gst_object_unref(bus);
  if (message == nullptr) {
    return false;
  }

  if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
    GError* error = nullptr;
    gchar* debug = nullptr;
    gst_message_parse_error(message, &error, &debug);
    std::cerr << "[GStreamer Capture JNI] Pipeline error";
    if (error != nullptr) {
      std::cerr << ": " << error->message;
      g_clear_error(&error);
    }
    if (debug != nullptr) {
      std::cerr << " (" << debug << ")";
      g_free(debug);
    }
    std::cerr << '\n';
  } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS) {
    std::cerr << "[GStreamer Capture JNI] Pipeline reached EOS\n";
  }

  gst_message_unref(message);
  return true;
}

void recordCaptureTiming(Capture* capture, GstBuffer* cpuBuffer, GstBuffer* mlBuffer) {
  if (capture == nullptr || capture->timingLogEvery <= 0 || capture->mlAppsink == nullptr) {
    return;
  }

  capture->timingFrames++;
  const GstClockTime cpuPts = cpuBuffer != nullptr ? GST_BUFFER_PTS(cpuBuffer) : GST_CLOCK_TIME_NONE;
  const GstClockTime mlPts = mlBuffer != nullptr ? GST_BUFFER_PTS(mlBuffer) : GST_CLOCK_TIME_NONE;
  if (GST_CLOCK_TIME_IS_VALID(cpuPts) && GST_CLOCK_TIME_IS_VALID(mlPts)) {
    const auto cpuSigned = static_cast<int64_t>(cpuPts);
    const auto mlSigned = static_cast<int64_t>(mlPts);
    const auto deltaNs = cpuSigned > mlSigned ? cpuSigned - mlSigned : mlSigned - cpuSigned;
    const double deltaMs = static_cast<double>(deltaNs) / 1000000.0;
    capture->timingMlMatched++;
    capture->timingMlPtsDeltaMsSum += deltaMs;
    capture->timingMlPtsDeltaMsMax = std::max(capture->timingMlPtsDeltaMsMax, deltaMs);
  } else {
    capture->timingMlMissing++;
  }

  if (capture->timingFrames >= capture->timingLogEvery) {
    const double avgDelta =
        capture->timingMlMatched > 0
            ? capture->timingMlPtsDeltaMsSum / static_cast<double>(capture->timingMlMatched)
            : 0.0;
    std::cerr << "[GStreamer Capture JNI] timing frames=" << capture->timingFrames
              << " mlMatched=" << capture->timingMlMatched
              << " mlMissing=" << capture->timingMlMissing
              << " strictCpuDrops=" << capture->timingStrictCpuDrops
              << " strictMlDrops=" << capture->timingStrictMlDrops
              << " avgCpuMlPtsDeltaMs=" << avgDelta
              << " maxCpuMlPtsDeltaMs=" << capture->timingMlPtsDeltaMsMax << '\n';
    capture->timingFrames = 0;
    capture->timingMlMatched = 0;
    capture->timingMlMissing = 0;
    capture->timingStrictCpuDrops = 0;
    capture->timingStrictMlDrops = 0;
    capture->timingMlPtsDeltaMsSum = 0.0;
    capture->timingMlPtsDeltaMsMax = 0.0;
  }
}

void readStats(Capture* capture, int64_t* out) {
  const CaptureStats& s = capture->stats;
  out[0] = s.sourceBuffers.load(std::memory_order_relaxed);
  out[1] = s.samplesPulled.load(std::memory_order_relaxed);
  out[2] = s.framesProduced.load(std::memory_order_relaxed);
  out[3] = s.framesDelivered.load(std::memory_order_relaxed);
  out[4] = s.framesReplaced.load(std::memory_order_relaxed);
  out[5] = s.decodeErrors.load(std::memory_order_relaxed);
  out[6] = s.corruptFrames.load(std::memory_order_relaxed);
  out[7] = s.convertNanosSum.load(std::memory_order_relaxed);
  out[8] = s.convertNanosMax.load(std::memory_order_relaxed);
  out[9] = s.ageSamples.load(std::memory_order_relaxed);
  out[10] = s.ageNanosSum.load(std::memory_order_relaxed);
  out[11] = s.ageNanosMax.load(std::memory_order_relaxed);
  out[12] = s.deviceDelayNanosSum.load(std::memory_order_relaxed);
  out[13] = s.sourcePtsInvalid.load(std::memory_order_relaxed);
  out[14] = s.captureTimeMissing.load(std::memory_order_relaxed);
  out[15] = s.ptsAheadOfClock.load(std::memory_order_relaxed);
}

// Prints where frames go. "source" is the count from the camera. "pulled" is the count that left
// the appsink. source - pulled is the count that the appsink dropped because the consumer was
// busy. "replaced" is the count of decoded frames that a newer decoded frame replaced.
void maybeLogStats(Capture* capture) {
  if (capture->timingLogEvery <= 0) {
    return;
  }
  const int64_t delivered = capture->stats.framesDelivered.load(std::memory_order_relaxed);
  if (delivered - capture->statsLogLast[3] < capture->timingLogEvery) {
    return;
  }

  int64_t now[kStatsFieldCount];
  readStats(capture, now);
  const int64_t nowNanos = monoNanos();
  const double seconds =
      capture->statsLogLastNanos > 0
          ? static_cast<double>(nowNanos - capture->statsLogLastNanos) / 1e9
          : 0.0;
  int64_t d[kStatsFieldCount];
  for (int i = 0; i < kStatsFieldCount; ++i) {
    d[i] = now[i] - capture->statsLogLast[i];
    capture->statsLogLast[i] = now[i];
  }
  capture->statsLogLastNanos = nowNanos;
  // Maximum values are not deltas. Reset them for the next window.
  const int64_t convertMax = capture->stats.convertNanosMax.exchange(0, std::memory_order_relaxed);
  const int64_t ageMax = capture->stats.ageNanosMax.exchange(0, std::memory_order_relaxed);
  capture->statsLogLast[8] = 0;
  capture->statsLogLast[11] = 0;

  if (seconds <= 0.0) {
    return;
  }

  char line[512];
  std::snprintf(
      line, sizeof(line),
      "[GStreamer Capture JNI] flow sourceFps=%.1f pulledFps=%.1f producedFps=%.1f "
      "deliveredFps=%.1f appsinkDrops=%lld replaced=%lld decodeErrors=%lld corrupt=%lld "
      "convertMsAvg=%.2f convertMsMax=%.2f ageMsAvg=%.2f ageMsMax=%.2f deviceDelayMsAvg=%.2f "
      "sourcePtsInvalid=%lld captureTimeMissing=%lld ptsAhead=%lld",
      static_cast<double>(d[0]) / seconds, static_cast<double>(d[1]) / seconds,
      static_cast<double>(d[2]) / seconds, static_cast<double>(d[3]) / seconds,
      static_cast<long long>(std::max<int64_t>(0, d[0] - d[1])), static_cast<long long>(d[4]),
      static_cast<long long>(d[5]), static_cast<long long>(d[6]),
      d[2] > 0 ? static_cast<double>(d[7]) / 1e6 / static_cast<double>(d[2]) : 0.0,
      static_cast<double>(convertMax) / 1e6,
      d[9] > 0 ? static_cast<double>(d[10]) / 1e6 / static_cast<double>(d[9]) : 0.0,
      static_cast<double>(ageMax) / 1e6,
      d[0] > 0 ? static_cast<double>(d[12]) / 1e6 / static_cast<double>(d[0]) : 0.0,
      static_cast<long long>(d[13]), static_cast<long long>(d[14]), static_cast<long long>(d[15]));
  std::cerr << line << '\n';
}

GstSample* pullMlSample(Capture* capture, GstClockTime cpuPts) {
  if (capture == nullptr || capture->mlAppsink == nullptr) {
    return nullptr;
  }

  if (!capture->strictMlPts || !GST_CLOCK_TIME_IS_VALID(cpuPts)) {
    return gst_app_sink_try_pull_sample(capture->mlAppsink, 5 * GST_MSECOND);
  }

  const int maxPulls = std::max(1, capture->strictMlMaxPulls);
  for (int pullIndex = 0; pullIndex < maxPulls; ++pullIndex) {
    GstSample* mlSample = capture->pendingMlSample;
    capture->pendingMlSample = nullptr;
    if (mlSample == nullptr) {
      mlSample = gst_app_sink_try_pull_sample(
          capture->mlAppsink, pullIndex == 0 ? 5 * GST_MSECOND : 0);
    }
    if (mlSample == nullptr) {
      return nullptr;
    }

    GstBuffer* mlBuffer = gst_sample_get_buffer(mlSample);
    const GstClockTime mlPts =
        mlBuffer != nullptr ? GST_BUFFER_PTS(mlBuffer) : GST_CLOCK_TIME_NONE;
    if (!GST_CLOCK_TIME_IS_VALID(mlPts)) {
      return mlSample;
    }
    if (mlPts == cpuPts) {
      return mlSample;
    }
    if (mlPts < cpuPts) {
      capture->timingStrictMlDrops++;
      gst_sample_unref(mlSample);
      continue;
    }

    // The ML branch has already advanced past this CPU frame. Keep that ML sample for the next
    // CPU frame and let the caller discard this CPU sample so boxes and ROI pixels stay same-frame.
    capture->pendingMlSample = mlSample;
    return nullptr;
  }

  return nullptr;
}

// ---------------------------------------------------------------------------------------------
// Pixel conversion
// ---------------------------------------------------------------------------------------------

// Copies the luma bytes of a packed 4:2:2 image (YUY2, YVYU: lumaOffset 0. UYVY, VYUY: lumaOffset
// 1) into a 1-channel image. This is a byte copy. The values are the values that the camera sent.
void extractPackedLuma(
    const uint8_t* src, int srcStride, int lumaOffset, uint8_t* dst, int dstStride, int width,
    int height) {
  for (int y = 0; y < height; ++y) {
    const uint8_t* s = src + static_cast<size_t>(y) * static_cast<size_t>(srcStride);
    uint8_t* d = dst + static_cast<size_t>(y) * static_cast<size_t>(dstStride);
    int x = 0;
#ifdef PV_HAVE_NEON
    for (; x + 16 <= width; x += 16) {
      const uint8x16x2_t pair = vld2q_u8(s + 2 * x);
      vst1q_u8(d + x, pair.val[lumaOffset]);
    }
#endif
    for (; x < width; ++x) {
      d[x] = s[2 * x + lumaOffset];
    }
  }
}

void copyPlane(
    const uint8_t* src, int srcStride, uint8_t* dst, int dstStride, int rowBytes, int height) {
  if (srcStride == dstStride && srcStride == rowBytes) {
    std::memcpy(dst, src, static_cast<size_t>(rowBytes) * static_cast<size_t>(height));
    return;
  }
  for (int y = 0; y < height; ++y) {
    std::memcpy(
        dst + static_cast<size_t>(y) * static_cast<size_t>(dstStride),
        src + static_cast<size_t>(y) * static_cast<size_t>(srcStride),
        static_cast<size_t>(rowBytes));
  }
}

enum class RawLayout {
  Unsupported,
  Interleaved,  // BGR, BGRx, BGRA, GRAY8: plane 0 is the image
  LumaPlane,  // planar or semi-planar YUV: plane 0 is an 8-bit luma image
  PackedLuma  // packed 4:2:2 YUV: luma is each second byte
};

struct RawFormat {
  RawLayout layout = RawLayout::Unsupported;
  int matType = 0;
  int channels = 0;
  int lumaOffset = 0;
};

RawFormat classifyRawFormat(const GstVideoInfo* info) {
  RawFormat result;
  const GstVideoFormat format = GST_VIDEO_INFO_FORMAT(info);
  switch (format) {
    case GST_VIDEO_FORMAT_BGR:
      result = {RawLayout::Interleaved, CV_8UC3, 3, 0};
      return result;
    case GST_VIDEO_FORMAT_BGRx:
    case GST_VIDEO_FORMAT_BGRA:
      result = {RawLayout::Interleaved, CV_8UC4, 4, 0};
      return result;
    case GST_VIDEO_FORMAT_GRAY8:
      result = {RawLayout::Interleaved, CV_8UC1, 1, 0};
      return result;
    default:
      break;
  }

  const GstVideoFormatInfo* finfo = info->finfo;
  // In a tiled format the stride value is a tile count, not a byte count. A copy with that value
  // would read outside the buffer.
  if (finfo == nullptr || GST_VIDEO_FORMAT_INFO_IS_TILED(finfo) ||
      !GST_VIDEO_FORMAT_INFO_IS_YUV(finfo) ||
      GST_VIDEO_FORMAT_INFO_DEPTH(finfo, 0) != 8 || GST_VIDEO_FORMAT_INFO_PLANE(finfo, 0) != 0) {
    return result;
  }

  const int planes = static_cast<int>(GST_VIDEO_FORMAT_INFO_N_PLANES(finfo));
  const int lumaPixelStride = GST_VIDEO_FORMAT_INFO_PSTRIDE(finfo, 0);
  const int lumaOffset = static_cast<int>(GST_VIDEO_FORMAT_INFO_POFFSET(finfo, 0));
  if (planes >= 2 && lumaPixelStride == 1 && lumaOffset == 0) {
    // I420, YV12, Y42B, Y444, Y41B, NV12, NV21, NV16, NV61, NV24 and equivalent formats.
    result = {RawLayout::LumaPlane, CV_8UC1, 1, 0};
  } else if (planes == 1 && lumaPixelStride == 2 && (lumaOffset == 0 || lumaOffset == 1)) {
    // YUY2, YVYU (luma first) and UYVY, VYUY (luma second).
    result = {RawLayout::PackedLuma, CV_8UC1, 1, lumaOffset};
  }
  return result;
}

void fillTimestamps(Capture* capture, FrameHolder* holder, GstBuffer* buffer) {
  if (buffer == nullptr) {
    return;
  }
  holder->captureMonoNanos = captureTimeFromBuffer(capture, buffer);
  holder->ptsNanos = gstTimeToJlong(GST_BUFFER_PTS(buffer));
  holder->dtsNanos = gstTimeToJlong(GST_BUFFER_DTS(buffer));
  holder->durationNanos = gstTimeToJlong(GST_BUFFER_DURATION(buffer));
  if (holder->captureMonoNanos < 0) {
    // Record the PTS-based estimate now, and the time of this estimate, so grab() can add the
    // time that the frame waits after this point.
    const jlong ptsAge = ptsEstimatedAgeNanos(capture, buffer);
    if (ptsAge >= 0) {
      holder->captureMonoNanos = monoNanos() - static_cast<int64_t>(ptsAge);
    }
    holder->ptsFallbackAgeNanos = ptsAge;
  }
}

// Builds a frame from a raw video sample. Takes ownership of `sample` in all cases.
FrameHolder* frameFromRawSample(Capture* capture, GstSample* sample, GstCaps* caps) {
  auto holder = std::make_unique<FrameHolder>();
  holder->sample = sample;
  holder->buffer = gst_sample_get_buffer(sample);

  GstVideoInfo videoInfo;
  if (holder->buffer == nullptr || caps == nullptr ||
      !gst_video_info_from_caps(&videoInfo, caps)) {
    discardFrameHolder(holder.release());
    return nullptr;
  }

  const RawFormat format = classifyRawFormat(&videoInfo);
  if (format.layout == RawLayout::Unsupported) {
    if (!capture->unsupportedFormatLogged) {
      std::cerr << "[GStreamer Capture JNI] Unsupported appsink video format: "
                << gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(&videoInfo)) << '\n';
      capture->unsupportedFormatLogged = true;
    }
    discardFrameHolder(holder.release());
    return nullptr;
  }

  // gst_video_frame_map() uses the GstVideoMeta of the buffer. Hardware decoders put the real
  // stride and plane offsets there, and these can be different from the values in the caps.
  if (!gst_video_frame_map(&holder->videoFrame, &videoInfo, holder->buffer, GST_MAP_READ)) {
    discardFrameHolder(holder.release());
    return nullptr;
  }
  holder->videoFrameMapped = true;

  const int width = GST_VIDEO_FRAME_WIDTH(&holder->videoFrame);
  const int height = GST_VIDEO_FRAME_HEIGHT(&holder->videoFrame);
  const int stride = GST_VIDEO_FRAME_PLANE_STRIDE(&holder->videoFrame, 0);
  const auto* planeData =
      static_cast<const uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&holder->videoFrame, 0));
  const int sourceRowBytes =
      format.layout == RawLayout::PackedLuma ? width * 2 : width * format.channels;
  if (width <= 0 || height <= 0 || planeData == nullptr || stride < sourceRowBytes) {
    discardFrameHolder(holder.release());
    return nullptr;
  }

  holder->sourceWidth = width;
  holder->sourceHeight = height;
  holder->channels = format.channels;
  fillTimestamps(capture, holder.get(), holder->buffer);

  // Java can write into the Mat in place (RotateImagePipe, the crosshair drawing). The mapping is
  // read-only, and for camera or decoder buffers (dmabuf) the pages are PROT_READ, so a write
  // stops the JVM. Wrap only interleaved frames in writable system memory, as from videoconvert.
  // All other frames are copied, and the GStreamer buffer goes back at once, so a frame in the
  // pipeline does not keep a camera buffer.
  bool wrapAllowed = capture->wrapMappedBuffer && format.layout == RawLayout::Interleaved;
  const guint memoryCount = wrapAllowed ? gst_buffer_n_memory(holder->buffer) : 0;
  for (guint i = 0; i < memoryCount && wrapAllowed; ++i) {
    GstMemory* memory = gst_buffer_peek_memory(holder->buffer, i);
    if (memory == nullptr || !gst_memory_is_type(memory, GST_ALLOCATOR_SYSMEM) ||
        GST_MEMORY_IS_READONLY(memory)) {
      wrapAllowed = false;
    }
  }
  const bool mustCopy = !wrapAllowed;
  if (!mustCopy) {
    // Zero-copy: the Mat points into the GStreamer buffer. The holder keeps the buffer mapped.
    holder->mat = new cv::Mat(
        height, width, format.matType, const_cast<uint8_t*>(planeData),
        static_cast<size_t>(stride));
    return holder.release();
  }

  const int64_t convertStart = monoNanos();
  const int dstStride = width * format.channels;
  const size_t dstSize = static_cast<size_t>(dstStride) * static_cast<size_t>(height);
  holder->pooled = capture->pool->acquire(dstSize);
  if (holder->pooled.data == nullptr) {
    discardFrameHolder(holder.release());
    return nullptr;
  }
  holder->pool = capture->pool;

  if (format.layout == RawLayout::PackedLuma) {
    extractPackedLuma(
        planeData, stride, format.lumaOffset, holder->pooled.data, dstStride, width, height);
  } else {
    copyPlane(planeData, stride, holder->pooled.data, dstStride, dstStride, height);
  }
  holder->mat = new cv::Mat(
      height, width, format.matType, holder->pooled.data, static_cast<size_t>(dstStride));

  // The pixels are now in our buffer. Give the GStreamer buffer back to its pool immediately.
  gst_video_frame_unmap(&holder->videoFrame);
  holder->videoFrameMapped = false;
  gst_sample_unref(holder->sample);
  holder->sample = nullptr;
  holder->buffer = nullptr;

  const int64_t convertNanos = monoNanos() - convertStart;
  capture->stats.convertNanosSum.fetch_add(convertNanos, std::memory_order_relaxed);
  atomicMax(capture->stats.convertNanosMax, convertNanos);
  return holder.release();
}

#ifdef PV_HAVE_LIBJPEG

// Standard Huffman tables (ITU-T T.81 Annex K.3). Motion-JPEG frames from UVC cameras frequently
// do not contain the tables. Recent libjpeg-turbo versions add them automatically. This copy is
// for other libjpeg versions.
const uint8_t kBitsDcLuminance[17] = {0, 0, 1, 5, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0};
const uint8_t kValDcLuminance[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t kBitsDcChrominance[17] = {0, 0, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0};
const uint8_t kValDcChrominance[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const uint8_t kBitsAcLuminance[17] = {0, 0, 2, 1, 3, 3, 2, 4, 3, 5, 5, 4, 4, 0, 0, 1, 0x7d};
const uint8_t kValAcLuminance[162] = {
    0x01, 0x02, 0x03, 0x00, 0x04, 0x11, 0x05, 0x12, 0x21, 0x31, 0x41, 0x06, 0x13, 0x51, 0x61,
    0x07, 0x22, 0x71, 0x14, 0x32, 0x81, 0x91, 0xa1, 0x08, 0x23, 0x42, 0xb1, 0xc1, 0x15, 0x52,
    0xd1, 0xf0, 0x24, 0x33, 0x62, 0x72, 0x82, 0x09, 0x0a, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x25,
    0x26, 0x27, 0x28, 0x29, 0x2a, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44, 0x45,
    0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63, 0x64,
    0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x83,
    0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97, 0x98, 0x99,
    0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4, 0xb5, 0xb6,
    0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca, 0xd2, 0xd3,
    0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe1, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7, 0xe8,
    0xe9, 0xea, 0xf1, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};
const uint8_t kBitsAcChrominance[17] = {0, 0, 2, 1, 2, 4, 4, 3, 4, 7, 5, 4, 4, 0, 1, 2, 0x77};
const uint8_t kValAcChrominance[162] = {
    0x00, 0x01, 0x02, 0x03, 0x11, 0x04, 0x05, 0x21, 0x31, 0x06, 0x12, 0x41, 0x51, 0x07, 0x61,
    0x71, 0x13, 0x22, 0x32, 0x81, 0x08, 0x14, 0x42, 0x91, 0xa1, 0xb1, 0xc1, 0x09, 0x23, 0x33,
    0x52, 0xf0, 0x15, 0x62, 0x72, 0xd1, 0x0a, 0x16, 0x24, 0x34, 0xe1, 0x25, 0xf1, 0x17, 0x18,
    0x19, 0x1a, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x43, 0x44,
    0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x63,
    0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a,
    0x82, 0x83, 0x84, 0x85, 0x86, 0x87, 0x88, 0x89, 0x8a, 0x92, 0x93, 0x94, 0x95, 0x96, 0x97,
    0x98, 0x99, 0x9a, 0xa2, 0xa3, 0xa4, 0xa5, 0xa6, 0xa7, 0xa8, 0xa9, 0xaa, 0xb2, 0xb3, 0xb4,
    0xb5, 0xb6, 0xb7, 0xb8, 0xb9, 0xba, 0xc2, 0xc3, 0xc4, 0xc5, 0xc6, 0xc7, 0xc8, 0xc9, 0xca,
    0xd2, 0xd3, 0xd4, 0xd5, 0xd6, 0xd7, 0xd8, 0xd9, 0xda, 0xe2, 0xe3, 0xe4, 0xe5, 0xe6, 0xe7,
    0xe8, 0xe9, 0xea, 0xf2, 0xf3, 0xf4, 0xf5, 0xf6, 0xf7, 0xf8, 0xf9, 0xfa};

void installHuffmanTable(
    j_decompress_ptr cinfo, JHUFF_TBL** slot, const uint8_t* bits, const uint8_t* values,
    int valueCount) {
  if (*slot != nullptr) {
    return;
  }
  *slot = jpeg_alloc_huff_table(reinterpret_cast<j_common_ptr>(cinfo));
  std::memcpy((*slot)->bits, bits, 17);
  std::memcpy((*slot)->huffval, values, static_cast<size_t>(valueCount));
}

void ensureStandardHuffmanTables(j_decompress_ptr cinfo) {
  installHuffmanTable(cinfo, &cinfo->dc_huff_tbl_ptrs[0], kBitsDcLuminance, kValDcLuminance, 12);
  installHuffmanTable(cinfo, &cinfo->ac_huff_tbl_ptrs[0], kBitsAcLuminance, kValAcLuminance, 162);
  installHuffmanTable(
      cinfo, &cinfo->dc_huff_tbl_ptrs[1], kBitsDcChrominance, kValDcChrominance, 12);
  installHuffmanTable(
      cinfo, &cinfo->ac_huff_tbl_ptrs[1], kBitsAcChrominance, kValAcChrominance, 162);
}

struct JpegErrorManager {
  jpeg_error_mgr pub;
  jmp_buf jump;
  // Fatal: libjpeg invented pixel data.
  bool dataDamaged = false;
  // The EOI marker is missing. All entropy data was there, else libjpeg also gives
  // JWRN_HIT_MARKER, which sets dataDamaged.
  bool prematureEof = false;
  // Bytes that libjpeg skipped before markers.
  unsigned extraneousBytesTotal = 0;
  // True while each skipped byte was before the EOI marker, after all MCUs were decoded.
  bool extraneousOnlyBeforeEoi = true;
  char message[JMSG_LENGTH_MAX] = {};
};

void jpegErrorExit(j_common_ptr cinfo) {
  auto* manager = reinterpret_cast<JpegErrorManager*>(cinfo->err);
  (*cinfo->err->format_message)(cinfo, manager->message);
  longjmp(manager->jump, 1);
}

// libjpeg reports damaged input as a warning and continues with invented pixel data. A frame
// with invented pixels must not go to the tag detector, so the warning type is recorded here.
// Some warnings are harmless: the pixels are decoded completely. Cameras can give these on each
// frame, so they must not reject the frame.
void jpegEmitMessage(j_common_ptr cinfo, int level) {
  if (level >= 0) {
    return;
  }
  auto* manager = reinterpret_cast<JpegErrorManager*>(cinfo->err);
  cinfo->err->num_warnings++;
  switch (cinfo->err->msg_code) {
    case JWRN_EXTRANEOUS_DATA:
      // msg_parm.i[0] is the count of skipped bytes, msg_parm.i[1] is the marker after them.
      manager->extraneousBytesTotal += static_cast<unsigned>(std::max(0, cinfo->err->msg_parm.i[0]));
      if (cinfo->err->msg_parm.i[1] != 0xD9) {
        manager->extraneousOnlyBeforeEoi = false;
      }
      break;
    case JWRN_JPEG_EOF:
      manager->prematureEof = true;
      break;
    case JWRN_NOT_SEQUENTIAL:
    case JWRN_JFIF_MAJOR:
    case JWRN_ADOBE_XFORM:
      // Harmless. The pixels are decoded completely.
      break;
    default:
      // JWRN_HIT_MARKER, JWRN_HUFF_BAD_CODE, JWRN_MUST_RESYNC, JWRN_BOGUS_PROGRESSION,
      // JWRN_TOO_MUCH_DATA and unknown codes: the pixels can be invented.
      manager->dataDamaged = true;
      break;
  }
}

struct JpegDecoder {
  jpeg_decompress_struct cinfo {};
  JpegErrorManager error {};
  bool created = false;

  JpegDecoder() {
    cinfo.err = jpeg_std_error(&error.pub);
    error.pub.error_exit = jpegErrorExit;
    error.pub.emit_message = jpegEmitMessage;
    if (setjmp(error.jump) == 0) {
      jpeg_create_decompress(&cinfo);
      created = true;
    }
  }

  ~JpegDecoder() {
    if (created) {
      jpeg_destroy_decompress(&cinfo);
    }
  }

  JpegDecoder(const JpegDecoder&) = delete;
  JpegDecoder& operator=(const JpegDecoder&) = delete;
};

// Suspect: the pixels are complete, but the frame had skipped bytes or no EOI marker. The caller
// decides if it accepts the frame.
enum class JpegResult { Ok, Error, Damaged, Suspect };

// Decodes one JPEG image into `out` (tight stride). When wantGray is true only the luma component
// is decoded: libjpeg skips the inverse DCT, the upsampling and the color conversion of the
// chroma components. The luma samples are the same samples that a full decode gives.
// dctMethod selects the inverse DCT: JDCT_ISLOW is the accurate method, JDCT_IFAST is the method
// of the GStreamer jpegdec element.
JpegResult decodeJpeg(
    JpegDecoder* decoder, BufferPool* pool, const uint8_t* data, size_t size, bool wantGray,
    J_DCT_METHOD dctMethod, PooledBuffer* out, int* outWidth, int* outHeight, int* outChannels,
    int rowPercent = 100) {
  // Volatile: these values change after setjmp() and are read after longjmp().
  PooledBuffer volatile buffer {};
  JSAMPROW rows[16];

  if (!decoder->created) {
    return JpegResult::Error;
  }

  jpeg_decompress_struct* cinfo = &decoder->cinfo;
  decoder->error.dataDamaged = false;
  decoder->error.prematureEof = false;
  decoder->error.extraneousBytesTotal = 0;
  decoder->error.extraneousOnlyBeforeEoi = true;
  decoder->error.pub.num_warnings = 0;

  if (setjmp(decoder->error.jump) != 0) {
    jpeg_abort_decompress(cinfo);
    PooledBuffer failed;
    failed.data = buffer.data;
    failed.capacity = buffer.capacity;
    pool->release(failed);
    return JpegResult::Error;
  }

  jpeg_mem_src(cinfo, const_cast<unsigned char*>(data), static_cast<unsigned long>(size));
  if (jpeg_read_header(cinfo, TRUE) != JPEG_HEADER_OK) {
    jpeg_abort_decompress(cinfo);
    return JpegResult::Error;
  }
  ensureStandardHuffmanTables(cinfo);

  const bool gray = wantGray || cinfo->num_components == 1;
  int channels = 1;
  bool swapRedBlue = false;
  if (gray) {
    cinfo->out_color_space = JCS_GRAYSCALE;
  } else {
#ifdef JCS_EXTENSIONS
    cinfo->out_color_space = JCS_EXT_BGR;
#else
    cinfo->out_color_space = JCS_RGB;
    swapRedBlue = true;
#endif
    channels = 3;
  }
  // The caller selects the inverse DCT. JDCT_ISLOW gives the reference pixel values. JDCT_IFAST
  // gives the same pixels as the jpegdec element of the old capture pipeline.
  cinfo->dct_method = dctMethod;
  cinfo->do_fancy_upsampling = TRUE;

  jpeg_start_decompress(cinfo);
  const int width = static_cast<int>(cinfo->output_width);
  const int height = static_cast<int>(cinfo->output_height);
  if (width <= 0 || height <= 0 || width > 16384 || height > 16384 ||
      cinfo->output_components != channels) {
    jpeg_abort_decompress(cinfo);
    return JpegResult::Error;
  }

  const size_t stride = static_cast<size_t>(width) * static_cast<size_t>(channels);
  {
    PooledBuffer acquired = pool->acquire(stride * static_cast<size_t>(height));
    buffer.data = acquired.data;
    buffer.capacity = acquired.capacity;
  }
  if (buffer.data == nullptr) {
    jpeg_abort_decompress(cinfo);
    return JpegResult::Error;
  }

  // rowPercent < 100 is only for the test aid: decode a part of the rows and stop.
  const JDIMENSION lastRow =
      rowPercent >= 100
          ? cinfo->output_height
          : static_cast<JDIMENSION>(
                static_cast<uint64_t>(cinfo->output_height) * static_cast<uint64_t>(rowPercent) / 100);
  while (cinfo->output_scanline < lastRow) {
    const int remaining = static_cast<int>(lastRow - cinfo->output_scanline);
    const int count = std::min(16, remaining);
    for (int i = 0; i < count; ++i) {
      rows[i] = buffer.data +
                (static_cast<size_t>(cinfo->output_scanline) + static_cast<size_t>(i)) * stride;
    }
    jpeg_read_scanlines(cinfo, rows, static_cast<JDIMENSION>(count));
  }
  if (lastRow < cinfo->output_height) {
    jpeg_abort_decompress(cinfo);
  } else {
    jpeg_finish_decompress(cinfo);
  }

  if (swapRedBlue) {
    uint8_t* p = buffer.data;
    const size_t pixels = static_cast<size_t>(width) * static_cast<size_t>(height);
    for (size_t i = 0; i < pixels; ++i, p += 3) {
      std::swap(p[0], p[2]);
    }
  }

  out->data = buffer.data;
  out->capacity = buffer.capacity;
  *outWidth = width;
  *outHeight = height;
  *outChannels = channels;
  if (decoder->error.dataDamaged) {
    return JpegResult::Damaged;
  }
  if (decoder->error.extraneousBytesTotal > 0 || decoder->error.prematureEof) {
    return JpegResult::Suspect;
  }
  return JpegResult::Ok;
}

// Builds a frame from a compressed JPEG sample. Takes ownership of `sample` in all cases.
FrameHolder* frameFromJpegSample(Capture* capture, JpegDecoder* decoder, GstSample* sample) {
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  GstMapInfo map {};
  if (buffer == nullptr || !gst_buffer_map(buffer, &map, GST_MAP_READ)) {
    gst_sample_unref(sample);
    capture->stats.decodeErrors.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }

  auto holder = std::make_unique<FrameHolder>();
  fillTimestamps(capture, holder.get(), buffer);

  const int64_t decodeStart = monoNanos();
  PooledBuffer pixels {};
  int width = 0;
  int height = 0;
  int channels = 0;
  // Read the method one time, so the test-only extra decode below uses the same method.
  const J_DCT_METHOD dctMethod =
      capture->jpegIdctAccurate.load(std::memory_order_relaxed) ? JDCT_ISLOW : JDCT_IFAST;
  const JpegResult result = decodeJpeg(
      decoder, capture->pool.get(), map.data, map.size, capture->outputMode == kOutputGray,
      dctMethod, &pixels, &width, &height, &channels);
  // Copy the warning fields now. The test-only extra decode below resets them.
  const bool prematureEof = decoder->error.prematureEof;
  const unsigned extraneousBytesTotal = decoder->error.extraneousBytesTotal;
  const bool extraneousOnlyBeforeEoi = decoder->error.extraneousOnlyBeforeEoi;

  // Test aid: extra decode work to simulate a slower processor. The pixels are discarded.
  for (int extra = testExtraDecodePercent.load(std::memory_order_relaxed); extra > 0;
       extra -= 100) {
    PooledBuffer scratch {};
    int w = 0;
    int h = 0;
    int c = 0;
    if (decodeJpeg(
            decoder, capture->pool.get(), map.data, map.size, capture->outputMode == kOutputGray,
            dctMethod, &scratch, &w, &h, &c, std::min(100, extra)) != JpegResult::Error) {
      capture->pool->release(scratch);
    }
  }
  gst_buffer_unmap(buffer, &map);
  gst_sample_unref(sample);

  if (result == JpegResult::Error) {
    capture->stats.decodeErrors.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }

  // A window of 600 decoded frames. A clean frame does not reset the counts, because a camera
  // that pads its frames gives the warning only on a part of the frames: the bit reader
  // frequently takes the padding without a warning.
  if (++capture->jpegWindowFrames >= 600 && !capture->tolerateExtraneousJpegBytes) {
    capture->jpegWindowFrames = 0;
    capture->jpegWindowSuspect = 0;
  }

  bool accept = result == JpegResult::Ok;
  if (result == JpegResult::Suspect) {
    if (capture->tolerateExtraneousJpegBytes) {
      accept = true;
    } else if (!prematureEof && extraneousOnlyBeforeEoi && extraneousBytesTotal <= 16) {
      // Alignment padding before EOI. All MCUs were decoded before the skipped bytes.
      accept = true;
    } else if (++capture->jpegWindowSuspect >= 30) {
      // Damage does not come this often. The camera adds padding or does not send EOI.
      capture->tolerateExtraneousJpegBytes = true;
      accept = true;
      std::cerr << "[GStreamer Capture JNI] Each JPEG frame has extraneous bytes. The camera "
                   "adds padding. These frames are now accepted.\n";
    }
  }
  if (!accept) {
    capture->pool->release(pixels);
    capture->stats.corruptFrames.fetch_add(1, std::memory_order_relaxed);
    return nullptr;
  }

  holder->pooled = pixels;
  holder->pool = capture->pool;
  holder->sourceWidth = width;
  holder->sourceHeight = height;
  holder->channels = channels;
  holder->mat = new cv::Mat(
      height, width, channels == 1 ? CV_8UC1 : CV_8UC3, pixels.data,
      static_cast<size_t>(width) * static_cast<size_t>(channels));

  const int64_t decodeNanos = monoNanos() - decodeStart;
  capture->stats.convertNanosSum.fetch_add(decodeNanos, std::memory_order_relaxed);
  atomicMax(capture->stats.convertNanosMax, decodeNanos);
  const int64_t previousAverage = capture->decodeNanosAverage.load(std::memory_order_relaxed);
  capture->decodeNanosAverage.store(
      previousAverage == 0 ? decodeNanos : (previousAverage * 7 + decodeNanos) / 8,
      std::memory_order_relaxed);
  return holder.release();
}

#endif  // PV_HAVE_LIBJPEG

bool capsAreJpeg(GstCaps* caps) {
  if (caps == nullptr || gst_caps_get_size(caps) == 0) {
    return false;
  }
  const GstStructure* structure = gst_caps_get_structure(caps, 0);
  return structure != nullptr && gst_structure_has_name(structure, "image/jpeg");
}

#ifdef PV_HAVE_LIBJPEG

// Worker for compressed input. The appsink keeps only the newest compressed frame. The worker
// decodes that frame and keeps only the newest decoded frame. Thus no stage can hold more than
// one frame, and the frame age has a fixed upper limit.
void workerLoop(Capture* capture) {
  pthread_setname_np(pthread_self(), "pv-jpeg-decode");
  pinDecodeWorkerFromEnvironment();
  applyRealtimePriorityToCurrentThread("JPEG decode");
  JpegDecoder decoder;

  while (!capture->workerStop.load()) {
    if (capture->decodePolicy != DecodePolicy::Newest) {
      std::unique_lock<std::mutex> lock(capture->slotMutex);
      const auto stopOrEmpty = [capture]() {
        return capture->latest == nullptr || capture->workerStop.load();
      };

      if (capture->decodePolicy == DecodePolicy::Demand) {
        capture->slotEmptied.wait(lock, stopOrEmpty);
      } else {
        // JustInTime. Wait until the consumer took the last frame. If the consumer does not
        // come for one camera frame period, decode a newer frame, so the frame in the slot does
        // not become old.
        int64_t framePeriod = capture->sourcePeriodNanosAverage.load(std::memory_order_relaxed);
        if (framePeriod <= 0) {
          framePeriod = 20000000LL;
        }
        capture->slotEmptied.wait_for(lock, std::chrono::nanoseconds(framePeriod), stopOrEmpty);

        // The consumer took the frame, and it did not have to wait for that frame. Thus the
        // consumer is slower than the camera. Start the next decode so that it ends a short time
        // before the consumer is expected to come back. Use the shortest of the last consumer
        // periods: a start that is too early only makes the frame a little older, but a start
        // that is too late makes the consumer wait.
        if (capture->latest == nullptr && !capture->lastTakeWaited && !capture->consumerWaiting &&
            capture->takeIntervalCount >= 2 && !capture->workerStop.load()) {
          int64_t consumerPeriod = capture->takeIntervalNanos[0];
          for (int i = 1; i < capture->takeIntervalCount; ++i) {
            consumerPeriod = std::min(consumerPeriod, capture->takeIntervalNanos[i]);
          }
          const int64_t decodeTime = capture->decodeNanosAverage.load(std::memory_order_relaxed);
          const int64_t margin = 500000LL + decodeTime / 8;
          const int64_t startAt = capture->lastTakeNanos + consumerPeriod - decodeTime - margin;
          const int64_t wait = startAt - monoNanos();
          if (wait > 0) {
            // If the consumer comes back before the start time, it sets consumerWaiting and
            // the decode starts immediately.
            capture->slotEmptied.wait_for(lock, std::chrono::nanoseconds(wait), [capture]() {
              return capture->consumerWaiting || capture->workerStop.load();
            });
          }
        }
      }
      if (capture->workerStop.load()) {
        break;
      }
    }

    GstSample* sample = gst_app_sink_try_pull_sample(capture->appsink, 50 * GST_MSECOND);
    if (sample == nullptr) {
      if (gst_app_sink_is_eos(capture->appsink)) {
        break;
      }
      continue;
    }
    capture->stats.samplesPulled.fetch_add(1, std::memory_order_relaxed);

    GstCaps* caps = gst_sample_get_caps(sample);
    FrameHolder* holder = capsAreJpeg(caps) ? frameFromJpegSample(capture, &decoder, sample)
                                            : frameFromRawSample(capture, sample, caps);
    if (holder == nullptr) {
      continue;
    }
    capture->stats.framesProduced.fetch_add(1, std::memory_order_relaxed);

    FrameHolder* replaced = nullptr;
    {
      std::lock_guard<std::mutex> lock(capture->slotMutex);
      replaced = capture->latest;
      capture->latest = holder;
    }
    capture->slotFilled.notify_one();
    if (replaced != nullptr) {
      capture->stats.framesReplaced.fetch_add(1, std::memory_order_relaxed);
      discardFrameHolder(replaced);
    }
  }

  {
    std::lock_guard<std::mutex> lock(capture->slotMutex);
    capture->workerEnded = true;
  }
  capture->slotFilled.notify_all();
}

#endif  // PV_HAVE_LIBJPEG

jlongArray frameToResult(JNIEnv* env, Capture* capture, FrameHolder* rawHolder) {
  std::unique_ptr<FrameHolder, void (*)(FrameHolder*)> holder(rawHolder, discardFrameHolder);

  jlong estimatedAge = -1;
  if (holder->captureMonoNanos >= 0) {
    estimatedAge =
        static_cast<jlong>(std::max<int64_t>(0, monoNanos() - holder->captureMonoNanos));
    capture->stats.ageSamples.fetch_add(1, std::memory_order_relaxed);
    capture->stats.ageNanosSum.fetch_add(estimatedAge, std::memory_order_relaxed);
    atomicMax(capture->stats.ageNanosMax, estimatedAge);
  } else {
    capture->stats.captureTimeMissing.fetch_add(1, std::memory_order_relaxed);
  }

  recordCaptureTiming(capture, holder->buffer, holder->mlBuffer);

  jlong values[kFieldCount] = {
      toHandle(holder.get()),
      toHandle(holder->mat),
      static_cast<jlong>(holder->sourceWidth),
      static_cast<jlong>(holder->sourceHeight),
      static_cast<jlong>(holder->channels),
      estimatedAge,
      holder->ptsNanos,
      holder->dtsNanos,
      holder->durationNanos,
      steadyNowNanos(),
      static_cast<jlong>(++capture->sequence),
      holder->mlBuffer != nullptr ? gstTimeToJlong(GST_BUFFER_PTS(holder->mlBuffer))
                                  : static_cast<jlong>(-1),
      holder->mlBuffer != nullptr ? gstTimeToJlong(GST_BUFFER_DTS(holder->mlBuffer))
                                  : static_cast<jlong>(-1),
      holder->mlBuffer != nullptr ? gstTimeToJlong(GST_BUFFER_DURATION(holder->mlBuffer))
                                  : static_cast<jlong>(-1)};

  jlongArray result = env->NewLongArray(kFieldCount);
  if (result == nullptr) {
    return nullptr;
  }
  env->SetLongArrayRegion(result, 0, kFieldCount, values);
  if (env->ExceptionCheck()) {
    return nullptr;
  }

  capture->stats.framesDelivered.fetch_add(1, std::memory_order_relaxed);
  maybeLogStats(capture);
  holder.release();
  return result;
}

int parseOutputMode(const std::string& value) {
  if (value == "gray" || value == "grey" || value == "gray8" || value == "mono") {
    return kOutputGray;
  }
  if (value == "bgr" || value == "color") {
    return kOutputBgr;
  }
  return kOutputAuto;
}

jlong createCapture(const std::string& pipelineText, int outputMode) {
  std::lock_guard<std::mutex> createLock(gstCreateMutex);
  ensureGstInit();

  if (pipelineText.empty()) {
    std::cerr << "[GStreamer Capture JNI] Empty pipeline string\n";
    return 0;
  }

  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(pipelineText.c_str(), &error);
  if (error != nullptr) {
    std::cerr << "[GStreamer Capture JNI] Failed to parse pipeline: " << error->message << '\n';
    g_clear_error(&error);
  }
  if (pipeline == nullptr) {
    return 0;
  }

  GstElement* sinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "pvappsink");
  if (sinkElement == nullptr || !GST_IS_APP_SINK(sinkElement)) {
    std::cerr << "[GStreamer Capture JNI] Pipeline must contain appsink name=pvappsink\n";
    if (sinkElement != nullptr) {
      gst_object_unref(sinkElement);
    }
    gst_object_unref(pipeline);
    return 0;
  }

  auto capture = std::make_unique<Capture>();
  capture->pipeline = pipeline;
  capture->appsink = GST_APP_SINK(sinkElement);
  capture->outputMode = outputMode;
  capture->wrapMappedBuffer = envFlagEnabled("PV_GSTREAMER_CAPTURE_WRAP_BUFFER");
  capture->strictMlPts = envFlagEnabled("PV_GSTREAMER_CAPTURE_STRICT_ML_PTS");
  capture->strictMlMaxPulls = std::max(1, envInt("PV_GSTREAMER_CAPTURE_STRICT_ML_MAX_PULLS", 4));
  capture->strictCpuMaxDrops = std::max(0, envInt("PV_GSTREAMER_CAPTURE_STRICT_CPU_MAX_DROPS", 2));
  capture->timingLogEvery = envInt("PV_GSTREAMER_CAPTURE_TIMING_LOG_EVERY", 0);
  const std::string decodePolicy = envLower("PV_GSTREAMER_CAPTURE_DECODE_POLICY");
  if (decodePolicy == "jit" || decodePolicy == "justintime" || decodePolicy == "just-in-time") {
    capture->decodePolicy = DecodePolicy::JustInTime;
  } else if (decodePolicy == "demand") {
    capture->decodePolicy = DecodePolicy::Demand;
  } else {
    capture->decodePolicy = DecodePolicy::Newest;
  }
  capture->tolerateExtraneousJpegBytes =
      envFlagEnabled("PV_GSTREAMER_CAPTURE_ALLOW_JPEG_EXTRA_BYTES");
  // PV_GSTREAMER_CAPTURE_JPEG_IDCT: "fast" gives the pixels of the jpegdec element (the old
  // pipeline). Each other value keeps the accurate method, which is nearer to the camera picture.
  capture->jpegIdctAccurate.store(envLower("PV_GSTREAMER_CAPTURE_JPEG_IDCT") != "fast");
  capture->captureTimeCaps = gst_caps_new_empty_simple("timestamp/x-photonvision-capture-mono");

  GstElement* mlSinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "pvmlsink");
  if (mlSinkElement != nullptr) {
    if (GST_IS_APP_SINK(mlSinkElement)) {
      capture->mlAppsink = GST_APP_SINK(mlSinkElement);
    } else {
      std::cerr << "[GStreamer Capture JNI] pvmlsink exists but is not an appsink\n";
      gst_object_unref(mlSinkElement);
    }
  }
  // One buffer, drop the older one. The appsink can never hold more than the newest frame.
  gst_app_sink_set_emit_signals(capture->appsink, false);
  gst_app_sink_set_drop(capture->appsink, true);
  gst_app_sink_set_max_buffers(capture->appsink, 1);
  g_object_set(G_OBJECT(capture->appsink), "enable-last-sample", FALSE, nullptr);
  if (capture->mlAppsink != nullptr) {
    gst_app_sink_set_emit_signals(capture->mlAppsink, false);
    gst_app_sink_set_drop(capture->mlAppsink, true);
    gst_app_sink_set_max_buffers(capture->mlAppsink, 1);
    g_object_set(G_OBJECT(capture->mlAppsink), "enable-last-sample", FALSE, nullptr);
  }

  installSourceProbe(capture.get());

  GstBus* bus = gst_element_get_bus(capture->pipeline);
  if (bus != nullptr) {
    gst_bus_set_sync_handler(bus, restorePriorityBusHandler, nullptr, nullptr);
    gst_object_unref(bus);
  }

  GstStateChangeReturn stateResult = gst_element_set_state(capture->pipeline, GST_STATE_PLAYING);
  if (stateResult == GST_STATE_CHANGE_FAILURE) {
    std::cerr << "[GStreamer Capture JNI] Failed to set pipeline PLAYING\n";
    logBusFailure(capture.get());
    destroyCapture(capture.release());
    return 0;
  }

  GstState state = GST_STATE_NULL;
  GstState pending = GST_STATE_NULL;
  stateResult = gst_element_get_state(capture->pipeline, &state, &pending, 3 * GST_SECOND);
  if (stateResult == GST_STATE_CHANGE_FAILURE) {
    std::cerr << "[GStreamer Capture JNI] Pipeline failed while entering PLAYING\n";
    logBusFailure(capture.get());
    destroyCapture(capture.release());
    return 0;
  }

  capture->clock = gst_element_get_clock(capture->pipeline);
  capture->baseTime = gst_element_get_base_time(capture->pipeline);
  capture->statsLogLastNanos = monoNanos();
  return toHandle(capture.release());
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_create(
    JNIEnv* env, jclass, jstring pipelineString) {
  return createCapture(
      toString(env, pipelineString), parseOutputMode(envLower("PV_GSTREAMER_CAPTURE_JPEG_OUTPUT")));
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_createWithOptions(
    JNIEnv* env, jclass, jstring pipelineString, jint outputMode) {
  const int mode =
      outputMode == kOutputGray || outputMode == kOutputBgr ? static_cast<int>(outputMode)
                                                            : kOutputAuto;
  return createCapture(toString(env, pipelineString), mode);
}

extern "C" JNIEXPORT jlong JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_getCapabilities(JNIEnv*, jclass) {
  jlong capabilities = kCapOptions | kCapStats | kCapLumaFormats | kCapThreadAffinity;
#ifdef PV_HAVE_LIBJPEG
  capabilities |= kCapJpegDecode | kCapJpegIdct;
#endif
  return capabilities;
}

extern "C" JNIEXPORT void JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_setJpegIdctAccurate(
    JNIEnv*, jclass, jlong capturePtr, jboolean accurate) {
  Capture* capture = fromHandle<Capture>(capturePtr);
  if (capture == nullptr) {
    return;
  }
  capture->jpegIdctAccurate.store(accurate == JNI_TRUE);
}

extern "C" JNIEXPORT void JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_setTestExtraDecodePercent(
    JNIEnv*, jclass, jint percent) {
  testExtraDecodePercent.store(std::max(0, static_cast<int>(percent)));
}

extern "C" JNIEXPORT jboolean JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_setCurrentThreadCpuMask(
    JNIEnv*, jclass, jlong cpuMask) {
  return setCurrentThreadCpuMask(static_cast<uint64_t>(cpuMask)) ? JNI_TRUE : JNI_FALSE;
}

extern "C" JNIEXPORT jboolean JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_hasElement(
    JNIEnv* env, jclass, jstring elementName) {
  ensureGstInit();
  const std::string name = toString(env, elementName);
  if (name.empty()) {
    return JNI_FALSE;
  }
  GstElementFactory* factory = gst_element_factory_find(name.c_str());
  if (factory == nullptr) {
    return JNI_FALSE;
  }
  gst_object_unref(factory);
  return JNI_TRUE;
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_getStats(
    JNIEnv* env, jclass, jlong capturePtr) {
  Capture* capture = fromHandle<Capture>(capturePtr);
  if (capture == nullptr) {
    return emptyResult(env);
  }

  int64_t stats[kStatsFieldCount];
  readStats(capture, stats);
  jlong values[kStatsFieldCount];
  for (int i = 0; i < kStatsFieldCount; ++i) {
    values[i] = static_cast<jlong>(stats[i]);
  }
  jlongArray result = env->NewLongArray(kStatsFieldCount);
  if (result != nullptr) {
    env->SetLongArrayRegion(result, 0, kStatsFieldCount, values);
  }
  return result;
}

extern "C" JNIEXPORT void JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_destroy(JNIEnv*, jclass, jlong capturePtr) {
  destroyCapture(fromHandle<Capture>(capturePtr));
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_grab(
    JNIEnv* env, jclass, jlong capturePtr, jlong timeoutMillis) {
  Capture* capture = fromHandle<Capture>(capturePtr);
  if (capture == nullptr || capture->appsink == nullptr) {
    return emptyResult(env);
  }

#ifdef PV_HAVE_LIBJPEG
  if (capture->workerRunning.load()) {
    // Compressed input: take the newest decoded frame from the worker.
    FrameHolder* holder = nullptr;
    {
      std::unique_lock<std::mutex> lock(capture->slotMutex);
      const auto ready = [capture]() { return capture->latest != nullptr || capture->workerEnded; };
      const bool mustWait = !ready();
      if (mustWait) {
        // Tell the worker that the consumer is here, so a delayed decode starts now.
        capture->consumerWaiting = true;
        capture->slotEmptied.notify_one();
      }
      if (timeoutMillis < 0) {
        capture->slotFilled.wait(lock, ready);
      } else {
        capture->slotFilled.wait_for(lock, std::chrono::milliseconds(timeoutMillis), ready);
      }
      capture->consumerWaiting = false;
      holder = capture->latest;
      capture->latest = nullptr;
      if (holder != nullptr) {
        const int64_t now = monoNanos();
        if (capture->lastTakeNanos > 0) {
          const int64_t interval = now - capture->lastTakeNanos;
          if (interval > 0 && interval < 1000000000LL) {
            capture->takeIntervalNanos[capture->takeIntervalNext] = interval;
            capture->takeIntervalNext = (capture->takeIntervalNext + 1) % 4;
            capture->takeIntervalCount = std::min(4, capture->takeIntervalCount + 1);
          } else {
            capture->takeIntervalCount = 0;
            capture->takeIntervalNext = 0;
          }
        }
        capture->lastTakeNanos = now;
        capture->lastTakeWaited = mustWait;
      }
    }
    capture->slotEmptied.notify_one();
    if (holder == nullptr) {
      logBusFailure(capture);
      return emptyResult(env);
    }
    return frameToResult(env, capture, holder);
  }
#endif

  const GstClockTime timeout =
      timeoutMillis < 0 ? GST_CLOCK_TIME_NONE : static_cast<GstClockTime>(timeoutMillis) * GST_MSECOND;
  GstSample* sample = nullptr;
  GstSample* mlSample = nullptr;
  const int maxCpuDrops =
      capture->strictMlPts && capture->mlAppsink != nullptr ? capture->strictCpuMaxDrops : 0;
  for (int cpuDrop = 0; cpuDrop <= maxCpuDrops; ++cpuDrop) {
    sample = gst_app_sink_try_pull_sample(capture->appsink, timeout);
    if (sample == nullptr) {
      logBusFailure(capture);
      return emptyResult(env);
    }
    capture->stats.samplesPulled.fetch_add(1, std::memory_order_relaxed);

    GstBuffer* sampleBuffer = gst_sample_get_buffer(sample);
    const GstClockTime cpuPts =
        sampleBuffer != nullptr ? GST_BUFFER_PTS(sampleBuffer) : GST_CLOCK_TIME_NONE;
    mlSample = pullMlSample(capture, cpuPts);
    if (!capture->strictMlPts || capture->mlAppsink == nullptr || mlSample != nullptr ||
        !GST_CLOCK_TIME_IS_VALID(cpuPts)) {
      break;
    }

    capture->timingStrictCpuDrops++;
    gst_sample_unref(sample);
    sample = nullptr;
  }
  if (sample == nullptr) {
    return emptyResult(env);
  }

  GstCaps* caps = gst_sample_get_caps(sample);
  FrameHolder* holder = nullptr;
  if (capsAreJpeg(caps)) {
#ifdef PV_HAVE_LIBJPEG
    // First compressed frame: decode it here, then start the worker for the next frames.
    {
      JpegDecoder decoder;
      holder = frameFromJpegSample(capture, &decoder, sample);
    }
    capture->workerStop.store(false);
    capture->workerRunning.store(true);
    capture->worker = std::thread(workerLoop, capture);
#else
    if (!capture->unsupportedFormatLogged) {
      std::cerr << "[GStreamer Capture JNI] The appsink gives image/jpeg, but this library was "
                   "built without libjpeg\n";
      capture->unsupportedFormatLogged = true;
    }
    gst_sample_unref(sample);
#endif
  } else {
    holder = frameFromRawSample(capture, sample, caps);
  }

  if (holder == nullptr) {
    if (mlSample != nullptr) {
      gst_sample_unref(mlSample);
    }
    return emptyResult(env);
  }
  capture->stats.framesProduced.fetch_add(1, std::memory_order_relaxed);

  if (capture->mlAppsink != nullptr) {
    holder->mlSample = mlSample;
    if (holder->mlSample != nullptr) {
      holder->mlBuffer = gst_sample_get_buffer(holder->mlSample);
      GstCaps* mlCaps = gst_sample_get_caps(holder->mlSample);
      GstVideoInfo mlVideoInfo;
      if (mlCaps != nullptr && gst_video_info_from_caps(&mlVideoInfo, mlCaps)) {
        holder->mlWidth = GST_VIDEO_INFO_WIDTH(&mlVideoInfo);
        holder->mlHeight = GST_VIDEO_INFO_HEIGHT(&mlVideoInfo);
      }
      if (holder->captureMonoNanos < 0 && holder->ptsNanos < 0) {
        fillTimestamps(capture, holder, holder->mlBuffer);
      }
    }
  } else if (mlSample != nullptr) {
    gst_sample_unref(mlSample);
  }

  return frameToResult(env, capture, holder);
}

extern "C" JNIEXPORT void JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_releaseFrame(JNIEnv*, jclass, jlong framePtr) {
  releaseFrameHolder(fromHandle<FrameHolder>(framePtr));
}
