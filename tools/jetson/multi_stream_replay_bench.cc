#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "absl/flags/flag.h"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "third_party/apriltag/apriltag.h"
#include "third_party/apriltag/common/workerpool.h"
#include "third_party/apriltag/tag36h11.h"

#include "aos/init.h"
#include "frc/orin/apriltag.h"

ABSL_FLAG(std::string, pipeline,
          "filesrc location=/tmp/ov9281_120.avi ! avidemux ! jpegdec ! "
          "videoconvert ! video/x-raw,format=GRAY8,width=1280,height=800 ! "
          "appsink drop=false sync=false",
          "GStreamer replay pipeline.");
ABSL_FLAG(int32_t, streams, 5, "Number of parallel replay streams.");
ABSL_FLAG(int32_t, width, 1280, "Frame width.");
ABSL_FLAG(int32_t, height, 800, "Frame height.");
ABSL_FLAG(int32_t, seconds, 2, "Benchmark duration.");
ABSL_FLAG(int32_t, threads, 3, "AprilTag CPU decode worker threads.");
ABSL_FLAG(int32_t, min_white_black_diff, 20,
          "Minimum black/white edge difference.");
ABSL_FLAG(int32_t, min_cluster_pixels, 25, "Minimum pixels in a cluster.");
ABSL_FLAG(bool, preload_frames, true,
          "Load the replay frames into RAM before timing detection.");
ABSL_FLAG(bool, serialize_detect, false,
          "Guard GpuDetector::Detect with a process-wide mutex.");
ABSL_FLAG(std::string, scheduler, "threaded",
          "threaded, round_robin, or shared_detector. round_robin cycles "
          "streams on one thread.");
ABSL_FLAG(bool, device_input_copy, false,
          "Copy MONO8 input into a per-detector CUDA buffer and pass it as "
          "device memory to GpuDetector::Detect.");

namespace {

using SteadyClock = std::chrono::steady_clock;
using SecondsDouble = std::chrono::duration<double>;
using MsDouble = std::chrono::duration<double, std::milli>;

struct Options {
  std::string pipeline;
  int streams;
  int width;
  int height;
  int seconds;
  int threads;
  int min_white_black_diff;
  int min_cluster_pixels;
  bool preload_frames;
  bool serialize_detect;
  std::string scheduler;
  bool device_input_copy;
};

Options GetOptions() {
  return Options{
      .pipeline = absl::GetFlag(FLAGS_pipeline),
      .streams = absl::GetFlag(FLAGS_streams),
      .width = absl::GetFlag(FLAGS_width),
      .height = absl::GetFlag(FLAGS_height),
      .seconds = absl::GetFlag(FLAGS_seconds),
      .threads = absl::GetFlag(FLAGS_threads),
      .min_white_black_diff = absl::GetFlag(FLAGS_min_white_black_diff),
      .min_cluster_pixels = absl::GetFlag(FLAGS_min_cluster_pixels),
      .preload_frames = absl::GetFlag(FLAGS_preload_frames),
      .serialize_detect = absl::GetFlag(FLAGS_serialize_detect),
      .scheduler = absl::GetFlag(FLAGS_scheduler),
      .device_input_copy = absl::GetFlag(FLAGS_device_input_copy),
  };
}

struct FamilyDeleter {
  void operator()(apriltag_family_t *family) const {
    if (family != nullptr) {
      tag36h11_destroy(family);
    }
  }
};

struct DetectorDeleter {
  void operator()(apriltag_detector_t *detector) const {
    if (detector != nullptr) {
      apriltag_detector_destroy(detector);
    }
  }
};

std::mutex &DetectMutex() {
  static std::mutex mutex;
  return mutex;
}

frc::apriltag::CameraMatrix IdentityCameraMatrix() {
  return frc::apriltag::CameraMatrix{
      .fx = 1.0,
      .cx = 1.0,
      .fy = 1.0,
      .cy = 1.0,
  };
}

frc::apriltag::DistCoeffs ZeroDistortion() {
  return frc::apriltag::DistCoeffs{
      .k1 = 0.0,
      .k2 = 0.0,
      .p1 = 0.0,
      .p2 = 0.0,
      .k3 = 0.0,
      .k4 = 0.0,
      .k5 = 0.0,
      .k6 = 0.0,
      .num_params = 5,
  };
}

struct WorkerStats {
  int64_t frames = 0;
  int64_t hits = 0;
  double capture_ms_total = 0.0;
  double detect_ms_total = 0.0;
  double worst_detect_ms = 0.0;
};

struct DetectorContext {
  DetectorContext(const Options &options)
      : width(options.width),
        height(options.height),
        family(tag36h11_create()),
        detector(apriltag_detector_create()) {
    apriltag_detector_add_family_bits(detector.get(), family.get(), 1);
    detector->nthreads = options.threads;
    detector->wp = workerpool_create(detector->nthreads);
    detector->quad_decimate = 2;
    detector->qtp.min_white_black_diff = options.min_white_black_diff;
    detector->qtp.min_cluster_pixels = options.min_cluster_pixels;
    gpu_detector = std::make_unique<frc::apriltag::GpuDetector>(
        options.width, options.height, detector.get(), IdentityCameraMatrix(),
        ZeroDistortion(), frc::vision::ImageFormat::MONO8);
    if (options.device_input_copy) {
      CHECK_CUDA(cudaMalloc(reinterpret_cast<void **>(&device_input),
                            width * height));
    }
  }

  ~DetectorContext() {
    if (device_input != nullptr) {
      CHECK_CUDA(cudaFree(device_input));
    }
  }

  const uint8_t *PrepareInput(const cv::Mat &frame) {
    if (device_input == nullptr) {
      return nullptr;
    }
    CHECK_CUDA(cudaMemcpy(device_input, frame.data, width * height,
                          cudaMemcpyHostToDevice));
    return device_input;
  }

  size_t width;
  size_t height;
  std::unique_ptr<apriltag_family_t, FamilyDeleter> family;
  std::unique_ptr<apriltag_detector_t, DetectorDeleter> detector;
  std::unique_ptr<frc::apriltag::GpuDetector> gpu_detector;
  uint8_t *device_input = nullptr;
};

std::vector<cv::Mat> LoadReplayFrames(const Options &options) {
  cv::VideoCapture capture;
  if (!capture.open(options.pipeline, cv::CAP_GSTREAMER)) {
    std::cerr << "failed to open preload pipeline\n";
    return {};
  }

  std::vector<cv::Mat> frames;
  cv::Mat frame;
  while (capture.read(frame)) {
    if (frame.cols != options.width || frame.rows != options.height ||
        frame.type() != CV_8UC1) {
      std::cerr << "unexpected preload frame " << frame.cols << "x"
                << frame.rows << " type=" << frame.type() << "\n";
      return {};
    }
    frames.push_back(frame.isContinuous() ? frame.clone()
                                          : frame.clone().clone());
  }
  return frames;
}

void RunWorker(const Options &options, int worker_id,
               const std::vector<cv::Mat> *preloaded_frames,
               std::atomic<bool> *start_flag, WorkerStats *stats) {
  CHECK_CUDA(cudaSetDevice(0));
  cv::VideoCapture capture;
  if (preloaded_frames == nullptr &&
      !capture.open(options.pipeline, cv::CAP_GSTREAMER)) {
    std::cerr << "worker " << worker_id << " failed to open pipeline\n";
    return;
  }

  DetectorContext context(options);

  cv::Mat captured_frame;
  size_t frame_index = 0;
  while (!start_flag->load(std::memory_order_acquire)) {
    std::this_thread::yield();
  }

  const auto start = SteadyClock::now();
  while (SecondsDouble(SteadyClock::now() - start).count() < options.seconds) {
    const auto capture_start = SteadyClock::now();
    const cv::Mat *frame = nullptr;
    if (preloaded_frames != nullptr) {
      frame = &(*preloaded_frames)[frame_index++ % preloaded_frames->size()];
    } else {
      if (!capture.read(captured_frame)) {
        break;
      }
      frame = &captured_frame;
    }
    const auto capture_end = SteadyClock::now();
    if (frame->cols != options.width || frame->rows != options.height ||
        frame->type() != CV_8UC1) {
      std::cerr << "worker " << worker_id << " unexpected frame "
                << frame->cols << "x" << frame->rows
                << " type=" << frame->type() << "\n";
      break;
    }
    if (!frame->isContinuous()) {
      captured_frame = frame->clone();
      frame = &captured_frame;
    }

    const auto detect_start = SteadyClock::now();
    const uint8_t *device_input = context.PrepareInput(*frame);
    if (options.serialize_detect) {
      std::lock_guard<std::mutex> lock(DetectMutex());
      context.gpu_detector->Detect(frame->data, device_input);
    } else {
      context.gpu_detector->Detect(frame->data, device_input);
    }
    const auto detect_end = SteadyClock::now();

    const double capture_ms = MsDouble(capture_end - capture_start).count();
    const double detect_ms = MsDouble(detect_end - detect_start).count();
    ++stats->frames;
    stats->capture_ms_total += capture_ms;
    stats->detect_ms_total += detect_ms;
    stats->worst_detect_ms = std::max(stats->worst_detect_ms, detect_ms);
    stats->hits += zarray_size(context.gpu_detector->Detections()) > 0 ? 1 : 0;
  }
}

void RunRoundRobin(const Options &options,
                   const std::vector<cv::Mat> &preloaded_frames,
                   std::vector<WorkerStats> *stats) {
  CHECK_CUDA(cudaSetDevice(0));
  std::vector<std::unique_ptr<DetectorContext>> contexts;
  contexts.reserve(options.streams);
  for (int i = 0; i < options.streams; ++i) {
    contexts.emplace_back(std::make_unique<DetectorContext>(options));
  }

  std::vector<size_t> frame_indices(options.streams, 0);
  const auto start = SteadyClock::now();
  while (SecondsDouble(SteadyClock::now() - start).count() < options.seconds) {
    for (int stream = 0; stream < options.streams; ++stream) {
      const auto capture_start = SteadyClock::now();
      const cv::Mat &frame =
          preloaded_frames[frame_indices[stream]++ % preloaded_frames.size()];
      const auto capture_end = SteadyClock::now();

      const auto detect_start = SteadyClock::now();
      const uint8_t *device_input = contexts[stream]->PrepareInput(frame);
      contexts[stream]->gpu_detector->Detect(frame.data, device_input);
      const auto detect_end = SteadyClock::now();

      const double capture_ms = MsDouble(capture_end - capture_start).count();
      const double detect_ms = MsDouble(detect_end - detect_start).count();
      WorkerStats &s = (*stats)[stream];
      ++s.frames;
      s.capture_ms_total += capture_ms;
      s.detect_ms_total += detect_ms;
      s.worst_detect_ms = std::max(s.worst_detect_ms, detect_ms);
      s.hits += zarray_size(contexts[stream]->gpu_detector->Detections()) > 0
                    ? 1
                    : 0;
    }
  }
}

void RunSharedDetector(const Options &options,
                       const std::vector<cv::Mat> &preloaded_frames,
                       std::vector<WorkerStats> *stats) {
  CHECK_CUDA(cudaSetDevice(0));
  DetectorContext context(options);

  std::vector<size_t> frame_indices(options.streams, 0);
  const auto start = SteadyClock::now();
  while (SecondsDouble(SteadyClock::now() - start).count() < options.seconds) {
    for (int stream = 0; stream < options.streams; ++stream) {
      const auto capture_start = SteadyClock::now();
      const cv::Mat &frame =
          preloaded_frames[frame_indices[stream]++ % preloaded_frames.size()];
      const auto capture_end = SteadyClock::now();

      const auto detect_start = SteadyClock::now();
      const uint8_t *device_input = context.PrepareInput(frame);
      context.gpu_detector->Detect(frame.data, device_input);
      const auto detect_end = SteadyClock::now();

      const double capture_ms = MsDouble(capture_end - capture_start).count();
      const double detect_ms = MsDouble(detect_end - detect_start).count();
      WorkerStats &s = (*stats)[stream];
      ++s.frames;
      s.capture_ms_total += capture_ms;
      s.detect_ms_total += detect_ms;
      s.worst_detect_ms = std::max(s.worst_detect_ms, detect_ms);
      s.hits += zarray_size(context.gpu_detector->Detections()) > 0 ? 1 : 0;
    }
  }
}

}  // namespace

int main(int argc, char **argv) {
  aos::InitGoogle(&argc, &argv);
  const Options options = GetOptions();
  std::vector<cv::Mat> preloaded_frames_storage;
  const std::vector<cv::Mat> *preloaded_frames = nullptr;
  if (options.preload_frames) {
    preloaded_frames_storage = LoadReplayFrames(options);
    if (preloaded_frames_storage.empty()) {
      std::cerr << "no preloaded frames available\n";
      return 1;
    }
    preloaded_frames = &preloaded_frames_storage;
    std::cout << "preloaded_frames=" << preloaded_frames_storage.size()
              << "\n";
  }

  std::vector<WorkerStats> stats(options.streams);

  if (options.scheduler == "round_robin" ||
      options.scheduler == "shared_detector") {
    if (preloaded_frames == nullptr) {
      std::cerr << options.scheduler << " requires --preload_frames=true\n";
      return 1;
    }
    const auto start = SteadyClock::now();
    if (options.scheduler == "round_robin") {
      RunRoundRobin(options, *preloaded_frames, &stats);
    } else {
      RunSharedDetector(options, *preloaded_frames, &stats);
    }
    const double elapsed = SecondsDouble(SteadyClock::now() - start).count();

    int64_t total_frames = 0;
    int64_t total_hits = 0;
    double total_capture_ms = 0.0;
    double total_detect_ms = 0.0;
    double worst_detect_ms = 0.0;
    for (int i = 0; i < options.streams; ++i) {
      const WorkerStats &s = stats[i];
      total_frames += s.frames;
      total_hits += s.hits;
      total_capture_ms += s.capture_ms_total;
      total_detect_ms += s.detect_ms_total;
      worst_detect_ms = std::max(worst_detect_ms, s.worst_detect_ms);
      std::cout << "stream=" << i << " frames=" << s.frames
                << " fps=" << (s.frames / elapsed)
                << " avg_capture_ms="
                << (s.frames == 0 ? 0.0 : s.capture_ms_total / s.frames)
                << " avg_detect_ms="
                << (s.frames == 0 ? 0.0 : s.detect_ms_total / s.frames)
                << " worst_detect_ms=" << s.worst_detect_ms << " hits="
                << s.hits << "/" << s.frames << "\n";
    }
    std::cout << "summary streams=" << options.streams
              << " frames=" << total_frames << " seconds=" << elapsed
              << " aggregate_fps=" << (total_frames / elapsed)
              << " avg_capture_ms="
              << (total_frames == 0 ? 0.0
                                    : total_capture_ms / total_frames)
              << " avg_detect_ms="
              << (total_frames == 0 ? 0.0 : total_detect_ms / total_frames)
              << " worst_detect_ms=" << worst_detect_ms << " hits="
              << total_hits << "/" << total_frames << "\n";
    return 0;
  }

  if (options.scheduler != "threaded") {
    std::cerr << "unsupported scheduler " << options.scheduler << "\n";
    return 1;
  }

  std::vector<std::thread> workers;
  std::atomic<bool> start_flag{false};

  workers.reserve(options.streams);
  for (int i = 0; i < options.streams; ++i) {
    workers.emplace_back(RunWorker, std::cref(options), i, preloaded_frames,
                         &start_flag, &stats[i]);
  }

  const auto start = SteadyClock::now();
  start_flag.store(true, std::memory_order_release);
  for (auto &worker : workers) {
    worker.join();
  }
  const double elapsed = SecondsDouble(SteadyClock::now() - start).count();

  int64_t total_frames = 0;
  int64_t total_hits = 0;
  double total_capture_ms = 0.0;
  double total_detect_ms = 0.0;
  double worst_detect_ms = 0.0;
  for (int i = 0; i < options.streams; ++i) {
    const WorkerStats &s = stats[i];
    total_frames += s.frames;
    total_hits += s.hits;
    total_capture_ms += s.capture_ms_total;
    total_detect_ms += s.detect_ms_total;
    worst_detect_ms = std::max(worst_detect_ms, s.worst_detect_ms);
    std::cout << "stream=" << i << " frames=" << s.frames
              << " fps=" << (s.frames / elapsed)
              << " avg_capture_ms="
              << (s.frames == 0 ? 0.0 : s.capture_ms_total / s.frames)
              << " avg_detect_ms="
              << (s.frames == 0 ? 0.0 : s.detect_ms_total / s.frames)
              << " worst_detect_ms=" << s.worst_detect_ms << " hits="
              << s.hits << "/" << s.frames << "\n";
  }
  std::cout << "summary streams=" << options.streams
            << " frames=" << total_frames << " seconds=" << elapsed
            << " aggregate_fps=" << (total_frames / elapsed)
            << " avg_capture_ms="
            << (total_frames == 0 ? 0.0 : total_capture_ms / total_frames)
            << " avg_detect_ms="
            << (total_frames == 0 ? 0.0 : total_detect_ms / total_frames)
            << " worst_detect_ms=" << worst_detect_ms << " hits="
            << total_hits << "/" << total_frames << "\n";
  return 0;
}
