#include <jni.h>

#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

constexpr int kFieldCount = 14;

std::once_flag gstInitFlag;
std::mutex gstCreateMutex;

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

bool envFlagEnabled(const char* name) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return false;
  }

  std::string value(raw);
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
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

jlong steadyNowNanos() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<jlong>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
}

struct Capture {
  GstElement* pipeline = nullptr;
  GstAppSink* appsink = nullptr;
  GstAppSink* mlAppsink = nullptr;
  GstSample* pendingMlSample = nullptr;
  GstClock* clock = nullptr;
  GstClockTime baseTime = GST_CLOCK_TIME_NONE;
  uint64_t sequence = 0;
  bool wrapMappedBuffer = false;
  bool strictMlPts = false;
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
};

struct FrameHolder {
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
};

void releaseFrameHolder(FrameHolder* holder) {
  if (holder == nullptr) {
    return;
  }

  // Java's OpenCV Mat wrapper owns this cv::Mat header after new Mat(nativeAddr). Deleting it here
  // races/double-frees with Mat.finalize(); this holder only owns the GStreamer sample/map lifetime.
  holder->mat = nullptr;

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

  delete holder;
}

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

void destroyCapture(Capture* capture) {
  if (capture == nullptr) {
    return;
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

  if (capture->pipeline != nullptr) {
    gst_object_unref(capture->pipeline);
    capture->pipeline = nullptr;
  }

  delete capture;
}

bool videoFormatToMatType(GstVideoFormat format, int* type, int* channels) {
  switch (format) {
    case GST_VIDEO_FORMAT_BGR:
      *type = CV_8UC3;
      *channels = 3;
      return true;
    case GST_VIDEO_FORMAT_BGRx:
    case GST_VIDEO_FORMAT_BGRA:
      *type = CV_8UC4;
      *channels = 4;
      return true;
    case GST_VIDEO_FORMAT_GRAY8:
      *type = CV_8UC1;
      *channels = 1;
      return true;
    default:
      return false;
  }
}

jlong estimatedAgeNanos(Capture* capture, GstBuffer* buffer) {
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
  if (!GST_CLOCK_TIME_IS_VALID(now) || now < absolutePts) {
    return 0;
  }

  return static_cast<jlong>(now - absolutePts);
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
  if (capture == nullptr || capture->timingLogEvery <= 0) {
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

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_create(
    JNIEnv* env, jclass, jstring pipelineString) {
  std::lock_guard<std::mutex> createLock(gstCreateMutex);

  std::call_once(gstInitFlag, []() {
    int argc = 0;
    char** argv = nullptr;
    gst_init(&argc, &argv);
  });

  const std::string pipelineText = toString(env, pipelineString);
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
  capture->wrapMappedBuffer = envFlagEnabled("PV_GSTREAMER_CAPTURE_WRAP_BUFFER");
  capture->strictMlPts = envFlagEnabled("PV_GSTREAMER_CAPTURE_STRICT_ML_PTS");
  capture->strictMlMaxPulls = std::max(1, envInt("PV_GSTREAMER_CAPTURE_STRICT_ML_MAX_PULLS", 4));
  capture->strictCpuMaxDrops = std::max(0, envInt("PV_GSTREAMER_CAPTURE_STRICT_CPU_MAX_DROPS", 2));
  capture->timingLogEvery = envInt("PV_GSTREAMER_CAPTURE_TIMING_LOG_EVERY", 0);

  GstElement* mlSinkElement = gst_bin_get_by_name(GST_BIN(pipeline), "pvmlsink");
  if (mlSinkElement != nullptr) {
    if (GST_IS_APP_SINK(mlSinkElement)) {
      capture->mlAppsink = GST_APP_SINK(mlSinkElement);
    } else {
      std::cerr << "[GStreamer Capture JNI] pvmlsink exists but is not an appsink\n";
      gst_object_unref(mlSinkElement);
    }
  }
  gst_app_sink_set_emit_signals(capture->appsink, false);
  gst_app_sink_set_drop(capture->appsink, true);
  gst_app_sink_set_max_buffers(capture->appsink, 1);
  if (capture->mlAppsink != nullptr) {
    gst_app_sink_set_emit_signals(capture->mlAppsink, false);
    gst_app_sink_set_drop(capture->mlAppsink, true);
    gst_app_sink_set_max_buffers(capture->mlAppsink, 1);
  }

  GstStateChangeReturn stateResult = gst_element_set_state(capture->pipeline, GST_STATE_PLAYING);
  if (stateResult == GST_STATE_CHANGE_FAILURE) {
    std::cerr << "[GStreamer Capture JNI] Failed to set pipeline PLAYING\n";
    destroyCapture(capture.release());
    return 0;
  }

  GstState state = GST_STATE_NULL;
  GstState pending = GST_STATE_NULL;
  stateResult = gst_element_get_state(capture->pipeline, &state, &pending, 3 * GST_SECOND);
  if (stateResult == GST_STATE_CHANGE_FAILURE) {
    std::cerr << "[GStreamer Capture JNI] Pipeline failed while entering PLAYING\n";
    destroyCapture(capture.release());
    return 0;
  }

  capture->clock = gst_element_get_clock(capture->pipeline);
  capture->baseTime = gst_element_get_base_time(capture->pipeline);
  return toHandle(capture.release());
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

  auto holder = std::make_unique<FrameHolder>();
  holder->sample = sample;
  holder->buffer = sample != nullptr ? gst_sample_get_buffer(sample) : nullptr;

  GstCaps* caps = gst_sample_get_caps(sample);
  GstVideoInfo videoInfo;
  if (caps == nullptr || !gst_video_info_from_caps(&videoInfo, caps)) {
    releaseFrameHolder(holder.release());
    return emptyResult(env);
  }

  int matType = 0;
  int channels = 0;
  const GstVideoFormat format = GST_VIDEO_INFO_FORMAT(&videoInfo);
  if (!videoFormatToMatType(format, &matType, &channels)) {
    std::cerr << "[GStreamer Capture JNI] Unsupported appsink video format: "
              << gst_video_format_to_string(format) << '\n';
    releaseFrameHolder(holder.release());
    return emptyResult(env);
  }

  const int width = GST_VIDEO_INFO_WIDTH(&videoInfo);
  const int height = GST_VIDEO_INFO_HEIGHT(&videoInfo);
  const int stride = GST_VIDEO_INFO_PLANE_STRIDE(&videoInfo, 0);
  holder->sourceWidth = width;
  holder->sourceHeight = height;

  if (width <= 0 || height <= 0 || stride <= 0) {
    releaseFrameHolder(holder.release());
    return emptyResult(env);
  }

  if (holder->buffer == nullptr || !gst_buffer_map(holder->buffer, &holder->map, GST_MAP_READ)) {
    releaseFrameHolder(holder.release());
    return emptyResult(env);
  }
  holder->mapped = true;

  const size_t minSize =
      static_cast<size_t>(std::max(0, height - 1)) * static_cast<size_t>(stride) +
      static_cast<size_t>(width) * static_cast<size_t>(channels);
  if (holder->map.size < minSize) {
    std::cerr << "[GStreamer Capture JNI] Buffer is smaller than expected for frame\n";
    releaseFrameHolder(holder.release());
    return emptyResult(env);
  }

  cv::Mat view(height, width, matType, holder->map.data, static_cast<size_t>(stride));
  if (capture->wrapMappedBuffer) {
    holder->mat = new cv::Mat(height, width, matType, holder->map.data, static_cast<size_t>(stride));
  } else {
    holder->mat = new cv::Mat();
    view.copyTo(*holder->mat);
    gst_buffer_unmap(holder->buffer, &holder->map);
    holder->mapped = false;
    gst_sample_unref(holder->sample);
    holder->sample = nullptr;
    holder->buffer = nullptr;
  }

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
    }
  }

  GstBuffer* timingBuffer = holder->buffer != nullptr ? holder->buffer : holder->mlBuffer;
  const jlong estimatedAge = estimatedAgeNanos(capture, timingBuffer);
  const jlong bufferPts =
      timingBuffer != nullptr ? gstTimeToJlong(GST_BUFFER_PTS(timingBuffer)) : static_cast<jlong>(-1);
  const jlong bufferDts =
      timingBuffer != nullptr ? gstTimeToJlong(GST_BUFFER_DTS(timingBuffer)) : static_cast<jlong>(-1);
  const jlong bufferDuration =
      timingBuffer != nullptr ? gstTimeToJlong(GST_BUFFER_DURATION(timingBuffer))
                              : static_cast<jlong>(-1);
  recordCaptureTiming(capture, holder->buffer, holder->mlBuffer);

  jlong values[kFieldCount] = {
      toHandle(holder.get()),
      toHandle(holder->mat),
      static_cast<jlong>(width),
      static_cast<jlong>(height),
      static_cast<jlong>(channels),
      estimatedAge,
      bufferPts,
      bufferDts,
      bufferDuration,
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
    releaseFrameHolder(holder.release());
    return nullptr;
  }
  env->SetLongArrayRegion(result, 0, kFieldCount, values);
  if (env->ExceptionCheck()) {
    releaseFrameHolder(holder.release());
    return nullptr;
  }

  holder.release();
  return result;
}

extern "C" JNIEXPORT void JNICALL
Java_org_photonvision_gstreamer_GStreamerCaptureJNI_releaseFrame(JNIEnv*, jclass, jlong framePtr) {
  releaseFrameHolder(fromHandle<FrameHolder>(framePtr));
}
