#include <jni.h>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_egl_interop.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include <gst/video/video.h>
#include <nvbufsurface.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>


#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

void launchTensorRTPreprocess(
    const uint8_t* srcDevice, int srcWidth, int srcHeight, int channels, size_t srcStepBytes,
    void* dstDevice, int dstWidth, int dstHeight, float scale, float dx, float dy, bool fp16,
    int channelOrder, cudaStream_t stream);

namespace {

class TrtLogger final : public nvinfer1::ILogger {
 public:
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      // Avoid depending on PhotonVision logging from this standalone JNI library.
      fprintf(stderr, "[TensorRT JNI] %s\n", msg);
    }
  }
};

struct Detection {
  float cx = 0.0F;
  float cy = 0.0F;
  float width = 0.0F;
  float height = 0.0F;
  int classId = 0;
  float confidence = 0.0F;
};

// Mirrors the private holder allocated by pv_gstreamer_capture_jni. Keep this POD-only and update
// both sides together; it lets TensorRT consume the NVMM side sample while the Java CVMat keeps the
// holder alive.
struct GStreamerFrameHolder {
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

struct GStreamerCudaResource {
  void* eglImage = nullptr;
  cudaGraphicsResource* cudaResource = nullptr;
  cudaEglFrame eglFrame {};
};

struct PreprocessGraphKey {
  const uint8_t* sourceDevice = nullptr;
  int srcWidth = 0;
  int srcHeight = 0;
  int channels = 0;
  size_t srcStepBytes = 0;
  float scale = 0.0F;
  float dx = 0.0F;
  float dy = 0.0F;
  int channelOrder = 0;

  bool operator==(const PreprocessGraphKey& other) const {
    return sourceDevice == other.sourceDevice && srcWidth == other.srcWidth &&
           srcHeight == other.srcHeight && channels == other.channels &&
           srcStepBytes == other.srcStepBytes && scale == other.scale && dx == other.dx &&
           dy == other.dy && channelOrder == other.channelOrder;
  }
};

struct PreprocessGraphKeyHash {
  size_t operator()(const PreprocessGraphKey& key) const {
    size_t seed = std::hash<const void*>{}(key.sourceDevice);
    auto mix = [&seed](size_t value) {
      seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
    };
    mix(std::hash<int>{}(key.srcWidth));
    mix(std::hash<int>{}(key.srcHeight));
    mix(std::hash<int>{}(key.channels));
    mix(std::hash<size_t>{}(key.srcStepBytes));
    mix(std::hash<float>{}(key.scale));
    mix(std::hash<float>{}(key.dx));
    mix(std::hash<float>{}(key.dy));
    mix(std::hash<int>{}(key.channelOrder));
    return seed;
  }
};

struct PreprocessCudaGraph {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
};

std::mutex gDetectorCreateMutex;
std::mutex gCudaGraphCaptureMutex;

template <typename T>
struct TrtDestroy {
  void operator()(T* ptr) const {
    delete ptr;
  }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDestroy<T>>;

void checkCuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

bool hasNvmmCaps(GstCaps* caps) {
  if (caps == nullptr) {
    return false;
  }
  for (guint i = 0; i < gst_caps_get_size(caps); ++i) {
    GstCapsFeatures* features = gst_caps_get_features(caps, i);
    if (features != nullptr && gst_caps_features_contains(features, "memory:NVMM")) {
      return true;
    }
  }
  return false;
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

size_t envSize(const char* name, size_t defaultValue) {
  const char* raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return defaultValue;
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(raw, &end, 10);
  if (end == raw || parsed > 1024ULL * 1024ULL * 1024ULL) {
    return defaultValue;
  }
  return static_cast<size_t>(parsed);
}

std::vector<char> readFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("Unable to open engine file: " + path);
  }
  file.seekg(0, std::ios::end);
  const auto size = file.tellg();
  if (size <= 0) {
    throw std::runtime_error("Engine file is empty: " + path);
  }
  file.seekg(0, std::ios::beg);
  std::vector<char> bytes(static_cast<size_t>(size));
  file.read(bytes.data(), size);
  if (!file) {
    throw std::runtime_error("Unable to read engine file: " + path);
  }
  return bytes;
}

int64_t volume(const nvinfer1::Dims& dims) {
  int64_t result = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) {
      throw std::runtime_error("Tensor shape is still dynamic after input shape was set");
    }
    result *= dims.d[i];
  }
  return result;
}

size_t elementSize(nvinfer1::DataType type) {
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return sizeof(float);
    case nvinfer1::DataType::kHALF:
      return sizeof(__half);
    default:
      throw std::runtime_error("Only FP32 and FP16 TensorRT tensors are supported");
  }
}

float iou(const Detection& a, const Detection& b) {
  const float ax1 = a.cx - a.width * 0.5F;
  const float ay1 = a.cy - a.height * 0.5F;
  const float ax2 = a.cx + a.width * 0.5F;
  const float ay2 = a.cy + a.height * 0.5F;
  const float bx1 = b.cx - b.width * 0.5F;
  const float by1 = b.cy - b.height * 0.5F;
  const float bx2 = b.cx + b.width * 0.5F;
  const float by2 = b.cy + b.height * 0.5F;

  const float ix1 = std::max(ax1, bx1);
  const float iy1 = std::max(ay1, by1);
  const float ix2 = std::min(ax2, bx2);
  const float iy2 = std::min(ay2, by2);
  const float iw = std::max(0.0F, ix2 - ix1);
  const float ih = std::max(0.0F, iy2 - iy1);
  const float intersection = iw * ih;
  const float unionArea = a.width * a.height + b.width * b.height - intersection;
  if (unionArea <= 0.0F) {
    return 0.0F;
  }
  return intersection / unionArea;
}

std::vector<Detection> nms(std::vector<Detection> detections, float nmsThreshold) {
  std::sort(detections.begin(), detections.end(), [](const auto& a, const auto& b) {
    return a.confidence > b.confidence;
  });

  std::vector<Detection> kept;
  std::vector<bool> removed(detections.size(), false);
  for (size_t i = 0; i < detections.size(); ++i) {
    if (removed[i]) {
      continue;
    }
    kept.push_back(detections[i]);
    for (size_t j = i + 1; j < detections.size(); ++j) {
      if (!removed[j] && detections[i].classId == detections[j].classId &&
          iou(detections[i], detections[j]) > nmsThreshold) {
        removed[j] = true;
      }
    }
  }
  return kept;
}

class Detector {
 public:
  Detector(const std::string& enginePath, int width, int height, int classes)
      : width_(width),
        height_(height),
        classes_(classes),
        gpuLetterbox_(envFlagEnabled("PV_TENSORRT_GPU_LETTERBOX")),
        cudaGraphEnabled_(envFlagEnabled("PV_TENSORRT_CUDA_GRAPH")),
        cudaGraphPreprocessEnabled_(
            cudaGraphEnabled_ && envFlagEnabled("PV_TENSORRT_CUDA_GRAPH_PREPROCESS")),
        gstreamerInferenceGraphEnabled_(
            cudaGraphEnabled_ && envFlagEnabled("PV_TENSORRT_GSTREAMER_INFERENCE_GRAPH")),
        gstreamerPreprocessGraphEnabled_(
            cudaGraphEnabled_ && envFlagEnabled("PV_TENSORRT_GSTREAMER_PREPROCESS_GRAPH")),
        cacheGStreamerEglResource_(envFlagEnabled("PV_TENSORRT_CACHE_EGL_RESOURCE")),
        timingLogEvery_(envInt("PV_TENSORRT_TIMING_LOG_EVERY", 0)),
        dumpOutputEvery_(envInt("PV_TENSORRT_DUMP_OUTPUT_EVERY", 0)),
        dumpOutputTopK_(std::max(1, envInt("PV_TENSORRT_DUMP_OUTPUT_TOPK", 8))),
        cudaGraphWarmupRemaining_(envInt("PV_TENSORRT_CUDA_GRAPH_WARMUP", 3)) {
    if (width_ <= 0 || height_ <= 0 || classes_ <= 0) {
      throw std::runtime_error("Width, height, and class count must be positive");
    }

    checkCuda(cudaSetDevice(0), "cudaSetDevice create");
    checkCuda(cudaFree(nullptr), "cudaFree init create");

    auto engineBytes = readFile(enginePath);
    initLibNvInferPlugins(&logger_, "");
    runtime_.reset(nvinfer1::createInferRuntime(logger_));
    if (!runtime_) {
      throw std::runtime_error("Failed to create TensorRT runtime");
    }
    engine_.reset(runtime_->deserializeCudaEngine(engineBytes.data(), engineBytes.size()));
    if (!engine_) {
      throw std::runtime_error("Failed to deserialize TensorRT engine");
    }
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
      throw std::runtime_error("Failed to create TensorRT execution context");
    }
    findTensorNames();
    configureShapes();
    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate");
    if (timingEnabled()) {
      checkCuda(cudaEventCreate(&timingStart_), "cudaEventCreate timing start");
      checkCuda(cudaEventCreate(&timingAfterUpload_), "cudaEventCreate timing after upload");
      checkCuda(cudaEventCreate(&timingAfterInfer_), "cudaEventCreate timing after infer");
      checkCuda(cudaEventCreate(&timingAfterOutputCopy_), "cudaEventCreate timing after output copy");
    }
    allocateBuffers();
    preallocateDeviceBuffersFromEnv();
  }

  ~Detector() {
    for (auto& entry : gstreamerCudaResources_) {
      if (entry.second.cudaResource != nullptr) {
        cudaGraphicsUnregisterResource(entry.second.cudaResource);
      }
    }
    gstreamerCudaResources_.clear();
    if (timingAfterOutputCopy_ != nullptr) {
      cudaEventDestroy(timingAfterOutputCopy_);
    }
    if (timingAfterInfer_ != nullptr) {
      cudaEventDestroy(timingAfterInfer_);
    }
    if (timingAfterUpload_ != nullptr) {
      cudaEventDestroy(timingAfterUpload_);
    }
    if (timingStart_ != nullptr) {
      cudaEventDestroy(timingStart_);
    }
    if (originalDevice_ != nullptr) {
      cudaFree(originalDevice_);
    }
    if (outputFloatPinned_) {
      cudaHostUnregister(outputFloat_.data());
    }
    if (outputHalfPinned_) {
      cudaHostUnregister(outputHalf_.data());
    }
    if (cudaGraphExec_ != nullptr) {
      cudaGraphExecDestroy(cudaGraphExec_);
    }
    if (cudaGraph_ != nullptr) {
      cudaGraphDestroy(cudaGraph_);
    }
    destroyPreprocessCudaGraph();
    destroyCachedPreprocessCudaGraphs();
    if (inputDevice_ != nullptr) {
      cudaFree(inputDevice_);
    }
    if (outputDevice_ != nullptr) {
      cudaFree(outputDevice_);
    }
    if (stream_ != nullptr) {
      cudaStreamDestroy(stream_);
    }
  }

  std::vector<Detection> detect(const cv::Mat& image, float confidenceThreshold, float nmsThreshold) {
    if (image.cols != width_ || image.rows != height_) {
      throw std::runtime_error("Request size does not match engine input size");
    }
    if (image.depth() != CV_8U) {
      throw std::runtime_error("Only CV_8U OpenCV Mats are supported");
    }
    const int channels = image.channels();
    if (channels != 1 && channels != 3 && channels != 4) {
      throw std::runtime_error("Only 1, 3, and 4 channel input images are supported");
    }

    const auto totalStart = std::chrono::steady_clock::now();
    recordTiming(timingStart_);
    uploadInput(image);
    recordTiming(timingAfterUpload_);
    enqueueInference();
    recordTiming(timingAfterInfer_);
    downloadOutput();
    recordTiming(timingAfterOutputCopy_);
    const auto decodeStart = std::chrono::steady_clock::now();
    auto detections = decodeYoloOutput(confidenceThreshold, nmsThreshold);
    const auto totalEnd = std::chrono::steady_clock::now();
    collectTiming(decodeStart, totalStart, totalEnd, detections.size());
    return detections;
  }

  std::vector<Detection> detectOriginal(
      const cv::Mat& image, float confidenceThreshold, float nmsThreshold) {
    if (image.cols <= 0 || image.rows <= 0) {
      throw std::runtime_error("Input image is empty");
    }
    if (image.depth() != CV_8U) {
      throw std::runtime_error("Only CV_8U OpenCV Mats are supported");
    }
    const int channels = image.channels();
    if (channels != 1 && channels != 3 && channels != 4) {
      throw std::runtime_error("Only 1, 3, and 4 channel input images are supported");
    }

    float dx = 0.0F;
    float dy = 0.0F;
    float scale = 1.0F;
    const auto totalStart = std::chrono::steady_clock::now();
    recordTiming(timingStart_);
    if (gpuLetterbox_) {
      const bool graphPreprocess = cudaGraphPreprocessEnabled_;
      scale = uploadOriginalInputCuda(image, dx, dy, !graphPreprocess);
      recordTiming(timingAfterUpload_);
      if (graphPreprocess) {
        enqueueGpuPreprocessAndInference();
      } else {
        enqueueInference();
      }
    } else {
      letterbox(image, dx, dy, scale);
      uploadLetterboxedInputCuda(letterboxBuffer_);
      recordTiming(timingAfterUpload_);
      enqueueInference();
    }
    recordTiming(timingAfterInfer_);
    downloadOutput();
    recordTiming(timingAfterOutputCopy_);

    const auto decodeStart = std::chrono::steady_clock::now();
    auto detections = decodeYoloOutput(confidenceThreshold, nmsThreshold);
    if (scale <= 0.0F) {
      const auto totalEnd = std::chrono::steady_clock::now();
      collectTiming(decodeStart, totalStart, totalEnd, detections.size());
      return detections;
    }

    for (auto& detection : detections) {
      detection.cx = (detection.cx - dx) / scale;
      detection.cy = (detection.cy - dy) / scale;
      detection.width /= scale;
      detection.height /= scale;
    }
    const auto totalEnd = std::chrono::steady_clock::now();
    collectTiming(decodeStart, totalStart, totalEnd, detections.size());
    return detections;
  }

  std::vector<Detection> detectGStreamerFrame(
      GStreamerFrameHolder* holder, float confidenceThreshold, float nmsThreshold) {
    const auto interopStart = std::chrono::steady_clock::now();
    GStreamerCudaMapping mapping = mapGStreamerCudaFrame(holder);
    NativeDetectTiming detectTiming;
    try {
      std::vector<Detection> detections = detectMappedCudaFrame(
          holder, mapping, confidenceThreshold, nmsThreshold, detectTiming);
      const auto cleanupStart = std::chrono::steady_clock::now();
      unmapGStreamerCudaFrame(holder, mapping);
      const auto cleanupEnd = std::chrono::steady_clock::now();
      const double interopMs =
          std::chrono::duration<double, std::milli>(detectTiming.computeStart - interopStart)
              .count();
      const double cleanupMs =
          std::chrono::duration<double, std::milli>(cleanupEnd - cleanupStart).count();
      collectTiming(
          detectTiming.decodeStart, detectTiming.totalStart, detectTiming.totalEnd,
          detections.size(), interopMs, cleanupMs, mapping.cacheHit);
      return detections;
    } catch (...) {
      unmapGStreamerCudaFrame(holder, mapping);
      throw;
    }
  }

  jdoubleArray detectionsToJava(JNIEnv* env, const std::vector<Detection>& detections) {
    javaDetectionScratch_.clear();
    javaDetectionScratch_.reserve(detections.size() * 6);
    for (const auto& detection : detections) {
      javaDetectionScratch_.push_back(detection.cx);
      javaDetectionScratch_.push_back(detection.cy);
      javaDetectionScratch_.push_back(detection.width);
      javaDetectionScratch_.push_back(detection.height);
      javaDetectionScratch_.push_back(static_cast<jdouble>(detection.classId));
      javaDetectionScratch_.push_back(detection.confidence);
    }

    jdoubleArray result =
        env->NewDoubleArray(static_cast<jsize>(javaDetectionScratch_.size()));
    if (result == nullptr) {
      return nullptr;
    }
    if (!javaDetectionScratch_.empty()) {
      env->SetDoubleArrayRegion(
          result, 0, static_cast<jsize>(javaDetectionScratch_.size()),
          javaDetectionScratch_.data());
    }
    return result;
  }

 private:
  struct GStreamerCudaMapping {
    GstMapInfo map {};
    NvBufSurface* surface = nullptr;
    cudaGraphicsResource* cudaResource = nullptr;
    cudaEglFrame eglFrame {};
    const uint8_t* sourceDevice = nullptr;
    int width = 0;
    int height = 0;
    size_t pitch = 0;
    bool gstMapped = false;
    bool eglMapped = false;
    bool unregisterResource = true;
    bool cacheHit = false;
  };

  struct NativeDetectTiming {
    std::chrono::steady_clock::time_point computeStart {};
    std::chrono::steady_clock::time_point totalStart {};
    std::chrono::steady_clock::time_point decodeStart {};
    std::chrono::steady_clock::time_point totalEnd {};
  };

  GStreamerCudaMapping mapGStreamerCudaFrame(GStreamerFrameHolder* holder) {
    if (holder == nullptr || holder->mlSample == nullptr || holder->mlBuffer == nullptr) {
      throw std::runtime_error("GStreamer frame does not contain an NVMM ML sample");
    }

    GstCaps* caps = gst_sample_get_caps(holder->mlSample);
    if (!hasNvmmCaps(caps)) {
      throw std::runtime_error("GStreamer ML sample is not memory:NVMM");
    }

    GStreamerCudaMapping mapping;
    if (!gst_buffer_map(holder->mlBuffer, &mapping.map, GST_MAP_READ)) {
      throw std::runtime_error("Failed to map GStreamer ML buffer");
    }
    mapping.gstMapped = true;

    try {
      if (mapping.map.size < sizeof(NvBufSurface)) {
        throw std::runtime_error("GStreamer ML buffer is too small for NvBufSurface");
      }

      mapping.surface = reinterpret_cast<NvBufSurface*>(mapping.map.data);
      if (mapping.surface->batchSize < 1 || mapping.surface->numFilled < 1 ||
          mapping.surface->surfaceList == nullptr) {
        throw std::runtime_error("GStreamer ML NvBufSurface is invalid");
      }
      if (mapping.surface->memType != NVBUF_MEM_SURFACE_ARRAY) {
        throw std::runtime_error("GStreamer ML NvBufSurface is not NVBUF_MEM_SURFACE_ARRAY");
      }

      checkCuda(cudaSetDevice(0), "cudaSetDevice");

      NvBufSurfaceParams& params = mapping.surface->surfaceList[0];
      const uint64_t bufferDesc = static_cast<uint64_t>(params.bufferDesc);
      if (cacheGStreamerEglResource_) {
        auto cached = gstreamerCudaResources_.find(bufferDesc);
        if (cached != gstreamerCudaResources_.end()) {
          mapping.cudaResource = cached->second.cudaResource;
          mapping.eglFrame = cached->second.eglFrame;
          mapping.unregisterResource = false;
          mapping.cacheHit = true;
        }
      }

      if (!mapping.cacheHit) {
        if (NvBufSurfaceMapEglImage(mapping.surface, 0) != 0 ||
            params.mappedAddr.eglImage == nullptr) {
          throw std::runtime_error("NvBufSurfaceMapEglImage failed");
        }
        mapping.eglMapped = true;

        checkCuda(
            cudaGraphicsEGLRegisterImage(
                &mapping.cudaResource, params.mappedAddr.eglImage,
                cudaGraphicsRegisterFlagsReadOnly),
            "cudaGraphicsEGLRegisterImage");
        checkCuda(
            cudaGraphicsResourceGetMappedEglFrame(&mapping.eglFrame, mapping.cudaResource, 0, 0),
            "cudaGraphicsResourceGetMappedEglFrame");

        if (cacheGStreamerEglResource_) {
          GStreamerCudaResource resource;
          resource.eglImage = params.mappedAddr.eglImage;
          resource.cudaResource = mapping.cudaResource;
          resource.eglFrame = mapping.eglFrame;
          gstreamerCudaResources_[bufferDesc] = resource;
          mapping.unregisterResource = false;
          mapping.eglMapped = false;
        }
      }

      if (mapping.eglFrame.frameType != cudaEglFrameTypePitch ||
          mapping.eglFrame.planeCount < 1) {
        throw std::runtime_error("CUDA EGL frame is not pitch-linear");
      }

      const cudaEglPlaneDesc& plane = mapping.eglFrame.planeDesc[0];
      const cudaPitchedPtr& pitch = mapping.eglFrame.frame.pPitch[0];
      if (pitch.ptr == nullptr || plane.width <= 0 || plane.height <= 0 || plane.pitch <= 0 ||
          plane.numChannels != 4) {
        throw std::runtime_error("CUDA EGL frame does not expose an RGBA pitch pointer");
      }

      mapping.sourceDevice = static_cast<const uint8_t*>(pitch.ptr);
      mapping.width = static_cast<int>(plane.width);
      mapping.height = static_cast<int>(plane.height);
      mapping.pitch = static_cast<size_t>(plane.pitch);
      return mapping;
    } catch (...) {
      unmapGStreamerCudaFrame(holder, mapping);
      throw;
    }
  }

  void unmapGStreamerCudaFrame(GStreamerFrameHolder* holder, GStreamerCudaMapping& mapping) {
    if (mapping.cudaResource != nullptr && mapping.unregisterResource) {
      cudaGraphicsUnregisterResource(mapping.cudaResource);
      mapping.cudaResource = nullptr;
    }
    if (mapping.eglMapped && mapping.surface != nullptr) {
      NvBufSurfaceUnMapEglImage(mapping.surface, 0);
      mapping.eglMapped = false;
    }
    if (mapping.gstMapped && holder != nullptr && holder->mlBuffer != nullptr) {
      gst_buffer_unmap(holder->mlBuffer, &mapping.map);
      mapping.gstMapped = false;
    }
  }

  std::vector<Detection> detectMappedCudaFrame(
      GStreamerFrameHolder* holder, const GStreamerCudaMapping& mapping,
      float confidenceThreshold, float nmsThreshold, NativeDetectTiming& timing) {
    float dx = 0.0F;
    float dy = 0.0F;
    const float scale = configureDevicePreprocess(
        mapping.sourceDevice, mapping.width, mapping.height, 4, mapping.pitch, 1, dx, dy);

    timing.computeStart = std::chrono::steady_clock::now();
    timing.totalStart = std::chrono::steady_clock::now();
    recordTiming(timingStart_);
    if (gstreamerPreprocessGraphEnabled_) {
      enqueueGpuPreprocessAndInference(true);
      recordTiming(timingAfterUpload_);
    } else {
      launchLastTensorRTPreprocess();
      recordTiming(timingAfterUpload_);
      if (gstreamerInferenceGraphEnabled_) {
        enqueueInference();
      } else {
        enqueueTensorRtDirect("TensorRT enqueueV3 failed for GStreamer frame");
      }
    }
    recordTiming(timingAfterInfer_);
    downloadOutput();
    recordTiming(timingAfterOutputCopy_);

    timing.decodeStart = std::chrono::steady_clock::now();
    auto detections = decodeYoloOutput(confidenceThreshold, nmsThreshold);
    if (scale > 0.0F) {
      for (auto& detection : detections) {
        detection.cx = (detection.cx - dx) / scale;
        detection.cy = (detection.cy - dy) / scale;
        detection.width /= scale;
        detection.height /= scale;
      }
    }
    if (holder != nullptr && holder->sourceWidth > 0 && holder->sourceHeight > 0 &&
        mapping.width > 0 && mapping.height > 0 &&
        (holder->sourceWidth != mapping.width || holder->sourceHeight != mapping.height)) {
      const float xScale = static_cast<float>(holder->sourceWidth) / static_cast<float>(mapping.width);
      const float yScale =
          static_cast<float>(holder->sourceHeight) / static_cast<float>(mapping.height);
      for (auto& detection : detections) {
        detection.cx *= xScale;
        detection.cy *= yScale;
        detection.width *= xScale;
        detection.height *= yScale;
      }
    }
    timing.totalEnd = std::chrono::steady_clock::now();
    return detections;
  }

  bool timingEnabled() const { return timingLogEvery_ > 0; }

  void enqueueInference() {
    if (!cudaGraphEnabled_) {
      enqueueTensorRtDirect("TensorRT enqueueV3 failed");
      return;
    }

    if (cudaGraphExec_ == nullptr && cudaGraphWarmupRemaining_ > 0) {
      enqueueTensorRtDirect("TensorRT enqueueV3 failed during CUDA graph warmup");
      checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize CUDA graph warmup");
      cudaGraphWarmupRemaining_--;
      return;
    }

    if (cudaGraphExec_ == nullptr) {
      captureCudaGraph();
    }

    checkCuda(cudaGraphLaunch(cudaGraphExec_, stream_), "cudaGraphLaunch TensorRT inference");
  }

  void captureCudaGraph() {
    std::lock_guard<std::mutex> lock(gCudaGraphCaptureMutex);
    if (cudaGraphExec_ != nullptr) {
      return;
    }

    checkCuda(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture");
    if (!context_->enqueueV3(stream_)) {
      cudaStreamEndCapture(stream_, &cudaGraph_);
      if (cudaGraph_ != nullptr) {
        cudaGraphDestroy(cudaGraph_);
        cudaGraph_ = nullptr;
      }
      throw std::runtime_error("TensorRT enqueueV3 failed during CUDA graph capture");
    }

    checkCuda(cudaStreamEndCapture(stream_, &cudaGraph_), "cudaStreamEndCapture");
    checkCuda(cudaGraphInstantiate(&cudaGraphExec_, cudaGraph_, 0), "cudaGraphInstantiate");
  }

  void enqueueTensorRtDirect(const char* errorMessage) {
    if (!context_->enqueueV3(stream_)) {
      throw std::runtime_error(errorMessage);
    }
  }

  void enqueueGpuPreprocessAndInference(bool cacheByPreprocessSource = false) {
    if (cacheByPreprocessSource) {
      const PreprocessGraphKey key = currentPreprocessGraphKey();
      auto cached = preprocessCudaGraphCache_.find(key);
      if (cached == preprocessCudaGraphCache_.end()) {
        cached = preprocessCudaGraphCache_.emplace(key, captureCurrentPreprocessCudaGraph()).first;
      }
      checkCuda(
          cudaGraphLaunch(cached->second.exec, stream_),
          "cudaGraphLaunch cached TensorRT preprocess/inference");
      return;
    }

    if (cudaGraphPreprocessExec_ == nullptr || !preprocessGraphMatchesLast()) {
      destroyPreprocessCudaGraph();
      capturePreprocessCudaGraph();
    }

    checkCuda(
        cudaGraphLaunch(cudaGraphPreprocessExec_, stream_),
        "cudaGraphLaunch TensorRT preprocess/inference");
  }

  bool preprocessGraphMatchesLast() const {
    return cudaGraphPreprocessSrcWidth_ == lastPreprocessSrcWidth_ &&
           cudaGraphPreprocessSrcHeight_ == lastPreprocessSrcHeight_ &&
           cudaGraphPreprocessChannels_ == lastPreprocessChannels_ &&
           cudaGraphPreprocessSrcStepBytes_ == lastPreprocessSrcStepBytes_ &&
           cudaGraphPreprocessScale_ == lastPreprocessScale_ &&
           cudaGraphPreprocessDx_ == lastPreprocessDx_ &&
           cudaGraphPreprocessDy_ == lastPreprocessDy_ &&
           cudaGraphPreprocessChannelOrder_ == lastPreprocessChannelOrder_ &&
           cudaGraphPreprocessSourceDevice_ == lastPreprocessDevice_;
  }

  void capturePreprocessCudaGraph() {
    std::lock_guard<std::mutex> lock(gCudaGraphCaptureMutex);
    if (cudaGraphPreprocessExec_ != nullptr && preprocessGraphMatchesLast()) {
      return;
    }

    checkCuda(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture preprocess/inference");
    launchLastTensorRTPreprocess();
    if (!context_->enqueueV3(stream_)) {
      cudaStreamEndCapture(stream_, &cudaGraphPreprocess_);
      if (cudaGraphPreprocess_ != nullptr) {
        cudaGraphDestroy(cudaGraphPreprocess_);
        cudaGraphPreprocess_ = nullptr;
      }
      throw std::runtime_error("TensorRT enqueueV3 failed during preprocess CUDA graph capture");
    }

    checkCuda(
        cudaStreamEndCapture(stream_, &cudaGraphPreprocess_),
        "cudaStreamEndCapture preprocess/inference");
    checkCuda(
        cudaGraphInstantiate(&cudaGraphPreprocessExec_, cudaGraphPreprocess_, 0),
        "cudaGraphInstantiate preprocess/inference");
    cudaGraphPreprocessSrcWidth_ = lastPreprocessSrcWidth_;
    cudaGraphPreprocessSrcHeight_ = lastPreprocessSrcHeight_;
    cudaGraphPreprocessChannels_ = lastPreprocessChannels_;
    cudaGraphPreprocessSrcStepBytes_ = lastPreprocessSrcStepBytes_;
    cudaGraphPreprocessScale_ = lastPreprocessScale_;
    cudaGraphPreprocessDx_ = lastPreprocessDx_;
    cudaGraphPreprocessDy_ = lastPreprocessDy_;
    cudaGraphPreprocessChannelOrder_ = lastPreprocessChannelOrder_;
    cudaGraphPreprocessSourceDevice_ = lastPreprocessDevice_;
  }

  void destroyPreprocessCudaGraph() {
    if (cudaGraphPreprocessExec_ != nullptr) {
      cudaGraphExecDestroy(cudaGraphPreprocessExec_);
      cudaGraphPreprocessExec_ = nullptr;
    }
    if (cudaGraphPreprocess_ != nullptr) {
      cudaGraphDestroy(cudaGraphPreprocess_);
      cudaGraphPreprocess_ = nullptr;
    }
    cudaGraphPreprocessSrcWidth_ = 0;
    cudaGraphPreprocessSrcHeight_ = 0;
    cudaGraphPreprocessChannels_ = 0;
    cudaGraphPreprocessSrcStepBytes_ = 0;
    cudaGraphPreprocessScale_ = 0.0F;
    cudaGraphPreprocessDx_ = 0.0F;
    cudaGraphPreprocessDy_ = 0.0F;
    cudaGraphPreprocessChannelOrder_ = 0;
    cudaGraphPreprocessSourceDevice_ = nullptr;
  }

  PreprocessGraphKey currentPreprocessGraphKey() const {
    return PreprocessGraphKey{
        lastPreprocessDevice_,       lastPreprocessSrcWidth_, lastPreprocessSrcHeight_,
        lastPreprocessChannels_,     lastPreprocessSrcStepBytes_,
        lastPreprocessScale_,        lastPreprocessDx_,
        lastPreprocessDy_,           lastPreprocessChannelOrder_};
  }

  PreprocessCudaGraph captureCurrentPreprocessCudaGraph() {
    std::lock_guard<std::mutex> lock(gCudaGraphCaptureMutex);
    PreprocessCudaGraph captured;

    checkCuda(
        cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal),
        "cudaStreamBeginCapture cached preprocess/inference");
    launchLastTensorRTPreprocess();
    if (!context_->enqueueV3(stream_)) {
      cudaStreamEndCapture(stream_, &captured.graph);
      if (captured.graph != nullptr) {
        cudaGraphDestroy(captured.graph);
      }
      throw std::runtime_error("TensorRT enqueueV3 failed during cached preprocess CUDA graph capture");
    }

    checkCuda(
        cudaStreamEndCapture(stream_, &captured.graph),
        "cudaStreamEndCapture cached preprocess/inference");
    checkCuda(
        cudaGraphInstantiate(&captured.exec, captured.graph, 0),
        "cudaGraphInstantiate cached preprocess/inference");
    return captured;
  }

  static void destroyPreprocessCudaGraph(PreprocessCudaGraph& graph) {
    if (graph.exec != nullptr) {
      cudaGraphExecDestroy(graph.exec);
      graph.exec = nullptr;
    }
    if (graph.graph != nullptr) {
      cudaGraphDestroy(graph.graph);
      graph.graph = nullptr;
    }
  }

  void destroyCachedPreprocessCudaGraphs() {
    for (auto& entry : preprocessCudaGraphCache_) {
      destroyPreprocessCudaGraph(entry.second);
    }
    preprocessCudaGraphCache_.clear();
  }

  void recordTiming(cudaEvent_t event) {
    if (event != nullptr) {
      checkCuda(cudaEventRecord(event, stream_), "cudaEventRecord timing");
    }
  }

  void collectTiming(
      std::chrono::steady_clock::time_point decodeStart,
      std::chrono::steady_clock::time_point totalStart,
      std::chrono::steady_clock::time_point totalEnd,
      size_t detections,
      double gstreamerInteropMs = 0.0,
      double gstreamerCleanupMs = 0.0,
      bool gstreamerCacheHit = false) {
    if (!timingEnabled()) {
      return;
    }

    checkCuda(cudaEventSynchronize(timingAfterOutputCopy_), "cudaEventSynchronize timing");
    float uploadPreprocessMs = 0.0F;
    float inferMs = 0.0F;
    float outputCopyMs = 0.0F;
    checkCuda(
        cudaEventElapsedTime(&uploadPreprocessMs, timingStart_, timingAfterUpload_),
        "cudaEventElapsedTime upload/preprocess");
    checkCuda(
        cudaEventElapsedTime(&inferMs, timingAfterUpload_, timingAfterInfer_),
        "cudaEventElapsedTime infer");
    checkCuda(
        cudaEventElapsedTime(&outputCopyMs, timingAfterInfer_, timingAfterOutputCopy_),
        "cudaEventElapsedTime output copy");

    const double cpuDecodeMs =
        std::chrono::duration<double, std::milli>(totalEnd - decodeStart).count();
    const double totalNativeMs =
        std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();

    timingFrames_++;
    timingUploadPreprocessMs_ += uploadPreprocessMs;
    timingInferMs_ += inferMs;
    timingOutputCopyMs_ += outputCopyMs;
    timingCpuDecodeMs_ += cpuDecodeMs;
    timingTotalNativeMs_ += totalNativeMs;
    timingGStreamerInteropMs_ += gstreamerInteropMs;
    timingGStreamerCleanupMs_ += gstreamerCleanupMs;
    timingGStreamerCacheHits_ += gstreamerCacheHit ? 1 : 0;
    timingDetections_ += static_cast<double>(detections);

    if (timingFrames_ >= timingLogEvery_) {
      const double inv = 1.0 / static_cast<double>(timingFrames_);
      fprintf(
          stderr,
          "[TensorRT JNI] timing frames=%d avgUploadPreprocessMs=%.3f avgInferMs=%.3f "
          "avgOutputCopyMs=%.3f avgCpuDecodeMs=%.3f avgTotalNativeMs=%.3f "
          "avgGstInteropMs=%.3f avgGstCleanupMs=%.3f gstCacheHitRate=%.3f "
          "avgDetections=%.3f gpuLetterbox=%s cudaGraph=%s cudaGraphPreprocess=%s "
          "gstPreprocessGraph=%s preprocessGraphCacheSize=%zu gstEglCache=%s "
          "gstEglCacheSize=%zu input=%dx%d\n",
          timingFrames_, timingUploadPreprocessMs_ * inv, timingInferMs_ * inv,
          timingOutputCopyMs_ * inv, timingCpuDecodeMs_ * inv, timingTotalNativeMs_ * inv,
          timingGStreamerInteropMs_ * inv, timingGStreamerCleanupMs_ * inv,
          static_cast<double>(timingGStreamerCacheHits_) * inv, timingDetections_ * inv,
          gpuLetterbox_ ? "true" : "false", cudaGraphEnabled_ ? "true" : "false",
          cudaGraphPreprocessEnabled_ ? "true" : "false",
          gstreamerPreprocessGraphEnabled_ ? "true" : "false",
          preprocessCudaGraphCache_.size(),
          cacheGStreamerEglResource_ ? "true" : "false", gstreamerCudaResources_.size(), width_,
          height_);
      timingFrames_ = 0;
      timingUploadPreprocessMs_ = 0.0;
      timingInferMs_ = 0.0;
      timingOutputCopyMs_ = 0.0;
      timingCpuDecodeMs_ = 0.0;
      timingTotalNativeMs_ = 0.0;
      timingGStreamerInteropMs_ = 0.0;
      timingGStreamerCleanupMs_ = 0.0;
      timingGStreamerCacheHits_ = 0;
      timingDetections_ = 0.0;
    }
  }

  float configureDevicePreprocess(
      const uint8_t* srcDevice, int srcWidth, int srcHeight, int channels, size_t srcStepBytes,
      int channelOrder, float& dx, float& dy) {
    const float scale = std::min(
        static_cast<float>(height_) / static_cast<float>(srcHeight),
        static_cast<float>(width_) / static_cast<float>(srcWidth));

    const int newWidth = std::max(1, static_cast<int>(std::round(srcWidth * scale)));
    const int newHeight = std::max(1, static_cast<int>(std::round(srcHeight * scale)));
    dx = static_cast<float>(width_ - newWidth) * 0.5F;
    dy = static_cast<float>(height_ - newHeight) * 0.5F;

    lastPreprocessDevice_ = srcDevice;
    lastPreprocessSrcWidth_ = srcWidth;
    lastPreprocessSrcHeight_ = srcHeight;
    lastPreprocessChannels_ = channels;
    lastPreprocessSrcStepBytes_ = srcStepBytes;
    lastPreprocessScale_ = scale;
    lastPreprocessDx_ = dx;
    lastPreprocessDy_ = dy;
    lastPreprocessChannelOrder_ = channelOrder;
    return scale;
  }

  float uploadOriginalInputCuda(const cv::Mat& image, float& dx, float& dy, bool launchPreprocess) {
    const float scale = std::min(
        static_cast<float>(height_) / static_cast<float>(image.rows),
        static_cast<float>(width_) / static_cast<float>(image.cols));

    const int newWidth = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    const int newHeight = std::max(1, static_cast<int>(std::round(image.rows * scale)));
    dx = static_cast<float>(width_ - newWidth) * 0.5F;
    dy = static_cast<float>(height_ - newHeight) * 0.5F;

    const cv::Mat* source = &image;
    if (!image.isContinuous()) {
      image.copyTo(contiguousOriginal_);
      source = &contiguousOriginal_;
    }

    const size_t sourceBytes = source->step[0] * static_cast<size_t>(source->rows);
    ensureOriginalDevice(sourceBytes);
    checkCuda(
        cudaMemcpyAsync(originalDevice_, source->data, sourceBytes, cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync original input");

    lastPreprocessDevice_ = static_cast<const uint8_t*>(originalDevice_);
    lastPreprocessSrcWidth_ = source->cols;
    lastPreprocessSrcHeight_ = source->rows;
    lastPreprocessChannels_ = source->channels();
    lastPreprocessSrcStepBytes_ = source->step[0];
    lastPreprocessScale_ = scale;
    lastPreprocessDx_ = dx;
    lastPreprocessDy_ = dy;
    lastPreprocessChannelOrder_ = 0;
    if (launchPreprocess) {
      launchLastTensorRTPreprocess();
    }
    return scale;
  }

  void launchLastTensorRTPreprocess() {
    launchTensorRTPreprocess(
        lastPreprocessDevice_, lastPreprocessSrcWidth_, lastPreprocessSrcHeight_,
        lastPreprocessChannels_, lastPreprocessSrcStepBytes_, inputDevice_, width_, height_,
        lastPreprocessScale_, lastPreprocessDx_, lastPreprocessDy_,
        inputType_ == nvinfer1::DataType::kHALF, lastPreprocessChannelOrder_, stream_);
  }

  void uploadLetterboxedInputCuda(const cv::Mat& image) {
    const cv::Mat* source = &image;
    if (!image.isContinuous()) {
      image.copyTo(contiguousOriginal_);
      source = &contiguousOriginal_;
    }

    const size_t sourceBytes = source->step[0] * static_cast<size_t>(source->rows);
    ensureOriginalDevice(sourceBytes);
    checkCuda(
        cudaMemcpyAsync(originalDevice_, source->data, sourceBytes, cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync letterboxed input");

    launchTensorRTPreprocess(
        static_cast<const uint8_t*>(originalDevice_), source->cols, source->rows, source->channels(),
        source->step[0], inputDevice_, width_, height_, 1.0F, 0.0F, 0.0F,
        inputType_ == nvinfer1::DataType::kHALF, 0, stream_);
  }

  void ensureOriginalDevice(size_t requiredBytes) {
    if (requiredBytes <= originalDeviceBytes_) {
      return;
    }
    if (originalDevice_ != nullptr) {
      cudaFree(originalDevice_);
      originalDevice_ = nullptr;
      originalDeviceBytes_ = 0;
      destroyPreprocessCudaGraph();
    }
    checkCuda(cudaMalloc(&originalDevice_, requiredBytes), "cudaMalloc original input");
    originalDeviceBytes_ = requiredBytes;
  }

  void preallocateDeviceBuffersFromEnv() {
    const size_t originalBytes = envSize("PV_TENSORRT_PREALLOC_ORIGINAL_BYTES", 0);
    if (originalBytes > 0) {
      ensureOriginalDevice(originalBytes);
    }
  }

  void letterbox(const cv::Mat& image, float& dx, float& dy, float& scale) {
    scale = std::min(
        static_cast<float>(height_) / static_cast<float>(image.rows),
        static_cast<float>(width_) / static_cast<float>(image.cols));

    const int newWidth = std::max(1, static_cast<int>(std::round(image.cols * scale)));
    const int newHeight = std::max(1, static_cast<int>(std::round(image.rows * scale)));
    const float dw = static_cast<float>(width_ - newWidth) * 0.5F;
    const float dh = static_cast<float>(height_ - newHeight) * 0.5F;
    const int left = static_cast<int>(std::round(dw - 0.1F));
    const int right = static_cast<int>(std::round(dw + 0.1F));
    const int top = static_cast<int>(std::round(dh - 0.1F));
    const int bottom = static_cast<int>(std::round(dh + 0.1F));

    if (newWidth != image.cols || newHeight != image.rows) {
      cv::resize(image, resizedBuffer_, cv::Size(newWidth, newHeight), 0.0, 0.0, cv::INTER_LINEAR);
    } else {
      image.copyTo(resizedBuffer_);
    }

    cv::copyMakeBorder(
        resizedBuffer_, letterboxBuffer_, top, bottom, left, right, cv::BORDER_CONSTANT,
        cv::Scalar(128, 128, 128, 128));

    dx = dw;
    dy = dh;
  }

  void findTensorNames() {
    for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
      const char* name = engine_->getIOTensorName(i);
      if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT && inputName_.empty()) {
        inputName_ = name;
      } else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT &&
                 outputName_.empty()) {
        outputName_ = name;
      }
    }
    if (inputName_.empty() || outputName_.empty()) {
      throw std::runtime_error("TensorRT engine must have at least one input and one output tensor");
    }
    inputType_ = engine_->getTensorDataType(inputName_.c_str());
    outputType_ = engine_->getTensorDataType(outputName_.c_str());
  }

  void configureShapes() {
    auto inputDims = engine_->getTensorShape(inputName_.c_str());
    if (inputDims.nbDims != 4) {
      throw std::runtime_error("Expected NCHW input tensor with 4 dimensions");
    }
    if (inputDims.d[0] == -1 || inputDims.d[2] == -1 || inputDims.d[3] == -1) {
      nvinfer1::Dims4 shape{1, 3, height_, width_};
      if (!context_->setInputShape(inputName_.c_str(), shape)) {
        throw std::runtime_error("Failed to set dynamic TensorRT input shape");
      }
    }

    auto concreteInput = context_->getTensorShape(inputName_.c_str());
    if (concreteInput.nbDims != 4 || concreteInput.d[0] != 1 || concreteInput.d[1] != 3 ||
        concreteInput.d[2] != height_ || concreteInput.d[3] != width_) {
      throw std::runtime_error("TensorRT engine input shape does not match requested 1x3xHxW");
    }

    outputDims_ = context_->getTensorShape(outputName_.c_str());
    outputElementCount_ = static_cast<size_t>(volume(outputDims_));
    if (outputElementCount_ == 0) {
      throw std::runtime_error("TensorRT output tensor is empty");
    }
  }

  void allocateBuffers() {
    inputElementCount_ = static_cast<size_t>(3) * width_ * height_;
    checkCuda(
        cudaMalloc(&inputDevice_, inputElementCount_ * elementSize(inputType_)), "cudaMalloc input");
    checkCuda(
        cudaMalloc(&outputDevice_, outputElementCount_ * elementSize(outputType_)), "cudaMalloc output");
    if (!context_->setTensorAddress(inputName_.c_str(), inputDevice_) ||
        !context_->setTensorAddress(outputName_.c_str(), outputDevice_)) {
      throw std::runtime_error("Failed to bind TensorRT tensor addresses");
    }

    if (inputType_ == nvinfer1::DataType::kFLOAT) {
      inputFloat_.resize(inputElementCount_);
    } else if (inputType_ == nvinfer1::DataType::kHALF) {
      inputHalf_.resize(inputElementCount_);
    } else {
      throw std::runtime_error("Unsupported TensorRT input data type");
    }

    outputFloat_.resize(outputElementCount_);
    if (outputType_ == nvinfer1::DataType::kHALF) {
      outputHalf_.resize(outputElementCount_);
    } else if (outputType_ != nvinfer1::DataType::kFLOAT) {
      throw std::runtime_error("Unsupported TensorRT output data type");
    }

    if (!outputFloat_.empty()) {
      checkCuda(
          cudaHostRegister(
              outputFloat_.data(), outputFloat_.size() * sizeof(float),
              cudaHostRegisterDefault),
          "cudaHostRegister outputFloat");
      outputFloatPinned_ = true;
    }
    if (!outputHalf_.empty()) {
      checkCuda(
          cudaHostRegister(
              outputHalf_.data(), outputHalf_.size() * sizeof(__half),
              cudaHostRegisterDefault),
          "cudaHostRegister outputHalf");
      outputHalfPinned_ = true;
    }
  }

  void uploadInput(const cv::Mat& image) {
    if (inputType_ == nvinfer1::DataType::kFLOAT) {
      fillInput<float>(image, inputFloat_.data(), [](float v) { return v; });
      checkCuda(
          cudaMemcpyAsync(
              inputDevice_, inputFloat_.data(), inputFloat_.size() * sizeof(float),
              cudaMemcpyHostToDevice, stream_),
          "cudaMemcpyAsync input");
      return;
    }

    fillInput<__half>(image, inputHalf_.data(), [](float v) { return __float2half(v); });
    checkCuda(
        cudaMemcpyAsync(
            inputDevice_, inputHalf_.data(), inputHalf_.size() * sizeof(__half),
            cudaMemcpyHostToDevice, stream_),
        "cudaMemcpyAsync half input");
  }

  template <typename T, typename Convert>
  void fillInput(const cv::Mat& image, T* dst, Convert convert) {
    const size_t plane = static_cast<size_t>(width_) * height_;
    const int channels = image.channels();
    for (int y = 0; y < height_; ++y) {
      const uint8_t* row = image.ptr<uint8_t>(y);
      for (int x = 0; x < width_; ++x) {
        const size_t pixelIndex = static_cast<size_t>(y) * width_ + x;
        const uint8_t* src = row + static_cast<size_t>(x) * channels;

        float red;
        float green;
        float blue;
        if (channels == 1) {
          red = green = blue = static_cast<float>(src[0]) / 255.0F;
        } else {
          blue = static_cast<float>(src[0]) / 255.0F;
          green = static_cast<float>(src[1]) / 255.0F;
          red = static_cast<float>(src[2]) / 255.0F;
        }

        dst[pixelIndex] = convert(red);
        dst[plane + pixelIndex] = convert(green);
        dst[2 * plane + pixelIndex] = convert(blue);
      }
    }
  }

  void downloadOutput() {
    if (outputType_ == nvinfer1::DataType::kFLOAT) {
      checkCuda(
          cudaMemcpyAsync(
              outputFloat_.data(), outputDevice_, outputFloat_.size() * sizeof(float),
              cudaMemcpyDeviceToHost, stream_),
          "cudaMemcpyAsync output");
      checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize output");
      return;
    }

    checkCuda(
        cudaMemcpyAsync(
            outputHalf_.data(), outputDevice_, outputHalf_.size() * sizeof(__half),
            cudaMemcpyDeviceToHost, stream_),
        "cudaMemcpyAsync half output");
    checkCuda(cudaStreamSynchronize(stream_), "cudaStreamSynchronize half output");
    std::transform(outputHalf_.begin(), outputHalf_.end(), outputFloat_.begin(), __half2float);
  }

  std::vector<Detection> decodeYoloOutput(float confidenceThreshold, float nmsThreshold) {
    if (outputDims_.nbDims != 3) {
      throw std::runtime_error("Expected YOLO output tensor with 3 dimensions");
    }

    const int d0 = outputDims_.d[0];
    const int d1 = outputDims_.d[1];
    const int d2 = outputDims_.d[2];
    if (d0 != 1) {
      throw std::runtime_error("Only batch size 1 is supported");
    }

    // Newer Ultralytics YOLO26 exports can produce an end-to-end detection tensor
    // shaped [1, 300, 6], with rows [x1, y1, x2, y2, confidence, class].
    // Convert it back to the cx/cy/width/height representation used by PV.
    if (d2 == 6 && d1 > 0) {
      dumpEndToEndRowsIfRequested(confidenceThreshold, nmsThreshold);
      std::vector<Detection> detections;
      detections.reserve(static_cast<size_t>(d1));
      for (int anchor = 0; anchor < d1; ++anchor) {
        const size_t base = static_cast<size_t>(anchor) * 6;
        const float confidence = outputFloat_[base + 4];
        if (confidence < confidenceThreshold) {
          continue;
        }

        const float x1 = outputFloat_[base];
        const float y1 = outputFloat_[base + 1];
        const float x2 = outputFloat_[base + 2];
        const float y2 = outputFloat_[base + 3];
        const float width = x2 - x1;
        const float height = y2 - y1;
        if (width <= 0.0F || height <= 0.0F) {
          continue;
        }

        Detection detection;
        detection.cx = (x1 + x2) * 0.5F;
        detection.cy = (y1 + y2) * 0.5F;
        detection.width = width;
        detection.height = height;
        detection.classId = static_cast<int>(std::round(outputFloat_[base + 5]));
        detection.confidence = confidence;
        detections.push_back(detection);
      }
      return nms(std::move(detections), nmsThreshold);
    }

    const bool channelsFirst = d1 <= 256 && d2 > d1;
    const int attrs = channelsFirst ? d1 : d2;
    const int anchors = channelsFirst ? d2 : d1;
    const int classes = std::min(classes_, attrs - 4);
    if (attrs < 5 || classes <= 0) {
      throw std::runtime_error("YOLO output tensor does not contain box and class scores");
    }

    auto at = [&](int attr, int anchor) -> float {
      if (channelsFirst) {
        return outputFloat_[static_cast<size_t>(attr) * anchors + anchor];
      }
      return outputFloat_[static_cast<size_t>(anchor) * attrs + attr];
    };

    std::vector<Detection> detections;
    for (int anchor = 0; anchor < anchors; ++anchor) {
      int bestClass = 0;
      float bestScore = at(4, anchor);
      for (int classIdx = 1; classIdx < classes; ++classIdx) {
        const float score = at(4 + classIdx, anchor);
        if (score > bestScore) {
          bestClass = classIdx;
          bestScore = score;
        }
      }

      if (bestScore < confidenceThreshold) {
        continue;
      }

      Detection detection;
      detection.cx = at(0, anchor);
      detection.cy = at(1, anchor);
      detection.width = at(2, anchor);
      detection.height = at(3, anchor);
      detection.classId = bestClass;
      detection.confidence = bestScore;
      if (detection.width > 0.0F && detection.height > 0.0F) {
        detections.push_back(detection);
      }
    }

    return nms(std::move(detections), nmsThreshold);
  }

  void dumpEndToEndRowsIfRequested(float confidenceThreshold, float nmsThreshold) {
    if (dumpOutputEvery_ <= 0) {
      return;
    }
    dumpOutputFrames_++;
    if (dumpOutputFrames_ < dumpOutputEvery_) {
      return;
    }
    dumpOutputFrames_ = 0;

    const int rows = outputDims_.d[1];
    std::vector<int> indices;
    indices.reserve(static_cast<size_t>(rows));
    for (int i = 0; i < rows; ++i) {
      indices.push_back(i);
    }
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
      return outputFloat_[static_cast<size_t>(a) * 6 + 4]
          > outputFloat_[static_cast<size_t>(b) * 6 + 4];
    });

    const int topK = std::min(dumpOutputTopK_, rows);
    fprintf(
        stderr,
        "[TensorRT JNI] e2e output topRows=%d rows=%d threshold=%.6f nms=%.3f\n",
        topK, rows, confidenceThreshold, nmsThreshold);
    for (int rank = 0; rank < topK; ++rank) {
      const int row = indices[rank];
      const size_t base = static_cast<size_t>(row) * 6;
      const float x1 = outputFloat_[base];
      const float y1 = outputFloat_[base + 1];
      const float x2 = outputFloat_[base + 2];
      const float y2 = outputFloat_[base + 3];
      const float score = outputFloat_[base + 4];
      const float cls = outputFloat_[base + 5];
      fprintf(
          stderr,
          "[TensorRT JNI] e2e row rank=%d index=%d x1=%.2f y1=%.2f x2=%.2f y2=%.2f "
          "w=%.2f h=%.2f score=%.6f cls=%.3f\n",
          rank, row, x1, y1, x2, y2, x2 - x1, y2 - y1, score, cls);
    }
  }

  int width_;
  int height_;
  int classes_;
  bool gpuLetterbox_;
  bool cudaGraphEnabled_;
  bool cudaGraphPreprocessEnabled_;
  bool gstreamerInferenceGraphEnabled_;
  bool gstreamerPreprocessGraphEnabled_;
  bool cacheGStreamerEglResource_;
  int timingLogEvery_;
  TrtLogger logger_;
  TrtUniquePtr<nvinfer1::IRuntime> runtime_;
  TrtUniquePtr<nvinfer1::ICudaEngine> engine_;
  TrtUniquePtr<nvinfer1::IExecutionContext> context_;
  std::string inputName_;
  std::string outputName_;
  nvinfer1::DataType inputType_ = nvinfer1::DataType::kFLOAT;
  nvinfer1::DataType outputType_ = nvinfer1::DataType::kFLOAT;
  nvinfer1::Dims outputDims_{};
  size_t inputElementCount_ = 0;
  size_t outputElementCount_ = 0;
  void* inputDevice_ = nullptr;
  void* outputDevice_ = nullptr;
  void* originalDevice_ = nullptr;
  size_t originalDeviceBytes_ = 0;
  const uint8_t* lastPreprocessDevice_ = nullptr;
  cudaStream_t stream_ = nullptr;
  cudaGraph_t cudaGraph_ = nullptr;
  cudaGraphExec_t cudaGraphExec_ = nullptr;
  cudaGraph_t cudaGraphPreprocess_ = nullptr;
  cudaGraphExec_t cudaGraphPreprocessExec_ = nullptr;
  cudaEvent_t timingStart_ = nullptr;
  cudaEvent_t timingAfterUpload_ = nullptr;
  cudaEvent_t timingAfterInfer_ = nullptr;
  cudaEvent_t timingAfterOutputCopy_ = nullptr;
  int timingFrames_ = 0;
  int dumpOutputEvery_ = 0;
  int dumpOutputTopK_ = 8;
  int dumpOutputFrames_ = 0;
  int cudaGraphWarmupRemaining_ = 0;
  double timingUploadPreprocessMs_ = 0.0;
  double timingInferMs_ = 0.0;
  double timingOutputCopyMs_ = 0.0;
  double timingCpuDecodeMs_ = 0.0;
  double timingTotalNativeMs_ = 0.0;
  double timingGStreamerInteropMs_ = 0.0;
  double timingGStreamerCleanupMs_ = 0.0;
  int timingGStreamerCacheHits_ = 0;
  double timingDetections_ = 0.0;
  int lastPreprocessSrcWidth_ = 0;
  int lastPreprocessSrcHeight_ = 0;
  int lastPreprocessChannels_ = 0;
  size_t lastPreprocessSrcStepBytes_ = 0;
  float lastPreprocessScale_ = 1.0F;
  float lastPreprocessDx_ = 0.0F;
  float lastPreprocessDy_ = 0.0F;
  int lastPreprocessChannelOrder_ = 0;
  int cudaGraphPreprocessSrcWidth_ = 0;
  int cudaGraphPreprocessSrcHeight_ = 0;
  int cudaGraphPreprocessChannels_ = 0;
  size_t cudaGraphPreprocessSrcStepBytes_ = 0;
  float cudaGraphPreprocessScale_ = 0.0F;
  float cudaGraphPreprocessDx_ = 0.0F;
  float cudaGraphPreprocessDy_ = 0.0F;
  int cudaGraphPreprocessChannelOrder_ = 0;
  const uint8_t* cudaGraphPreprocessSourceDevice_ = nullptr;
  std::vector<float> inputFloat_;
  std::vector<__half> inputHalf_;
  std::vector<float> outputFloat_;
  std::vector<__half> outputHalf_;
  std::vector<jdouble> javaDetectionScratch_;
  bool outputFloatPinned_ = false;
  bool outputHalfPinned_ = false;
  cv::Mat resizedBuffer_;
  cv::Mat letterboxBuffer_;
  cv::Mat contiguousOriginal_;
  std::unordered_map<uint64_t, GStreamerCudaResource> gstreamerCudaResources_;
  std::unordered_map<PreprocessGraphKey, PreprocessCudaGraph, PreprocessGraphKeyHash>
      preprocessCudaGraphCache_;
};

void throwRuntimeException(JNIEnv* env, const std::string& message) {
  jclass clazz = env->FindClass("java/lang/RuntimeException");
  if (clazz != nullptr) {
    env->ThrowNew(clazz, message.c_str());
  }
}

void throwUnsupportedOperation(JNIEnv* env, const std::string& message) {
  jclass clazz = env->FindClass("java/lang/UnsupportedOperationException");
  if (clazz != nullptr) {
    env->ThrowNew(clazz, message.c_str());
  }
}

Detector* fromHandle(jlong ptr) {
  if (ptr == 0) {
    throw std::runtime_error("TensorRT detector pointer is null");
  }
  return reinterpret_cast<Detector*>(ptr);
}

}  // namespace

extern "C" {

JNIEXPORT jlong JNICALL Java_org_photonvision_tensorrt_TensorRTJNI_create(
    JNIEnv* env, jclass, jstring enginePath, jint width, jint height, jint classes) {
  const char* enginePathChars = env->GetStringUTFChars(enginePath, nullptr);
  if (enginePathChars == nullptr) {
    return 0;
  }
  std::string enginePathString(enginePathChars);
  env->ReleaseStringUTFChars(enginePath, enginePathChars);

  try {
    std::lock_guard<std::mutex> lock(gDetectorCreateMutex);
    return reinterpret_cast<jlong>(
        new Detector(enginePathString, static_cast<int>(width), static_cast<int>(height),
                     static_cast<int>(classes)));
  } catch (const std::exception& ex) {
    throwRuntimeException(env, ex.what());
    return 0;
  }
}

JNIEXPORT void JNICALL Java_org_photonvision_tensorrt_TensorRTJNI_destroy(
    JNIEnv*, jclass, jlong ptr) {
  delete reinterpret_cast<Detector*>(ptr);
}

JNIEXPORT jdoubleArray JNICALL Java_org_photonvision_tensorrt_TensorRTJNI_detect(
    JNIEnv* env, jclass, jlong ptr, jlong matNativeObjAddr, jdouble nmsThreshold,
    jdouble confidenceThreshold) {
  try {
    Detector* detector = fromHandle(ptr);
    const auto* mat = reinterpret_cast<cv::Mat*>(matNativeObjAddr);
    if (mat == nullptr) {
      throw std::runtime_error("OpenCV Mat pointer is null");
    }

    std::vector<Detection> detections =
        detector->detect(*mat, static_cast<float>(confidenceThreshold), static_cast<float>(nmsThreshold));
    return detector->detectionsToJava(env, detections);
  } catch (const std::exception& ex) {
    throwRuntimeException(env, ex.what());
    return nullptr;
  }
}

JNIEXPORT jdoubleArray JNICALL Java_org_photonvision_tensorrt_TensorRTJNI_detectOriginal(
    JNIEnv* env, jclass, jlong ptr, jlong matNativeObjAddr, jdouble nmsThreshold,
    jdouble confidenceThreshold) {
  try {
    Detector* detector = fromHandle(ptr);
    const auto* mat = reinterpret_cast<cv::Mat*>(matNativeObjAddr);
    if (mat == nullptr) {
      throw std::runtime_error("OpenCV Mat pointer is null");
    }

    std::vector<Detection> detections =
        detector->detectOriginal(*mat, static_cast<float>(confidenceThreshold),
                                 static_cast<float>(nmsThreshold));
    return detector->detectionsToJava(env, detections);
  } catch (const std::exception& ex) {
    throwRuntimeException(env, ex.what());
    return nullptr;
  }
}

JNIEXPORT jdoubleArray JNICALL Java_org_photonvision_tensorrt_TensorRTJNI_detectGStreamerFrame(
    JNIEnv* env, jclass, jlong ptr, jlong nativeFrameHandle, jdouble nmsThreshold,
    jdouble confidenceThreshold) {
  try {
    Detector* detector = fromHandle(ptr);
    auto* holder = reinterpret_cast<GStreamerFrameHolder*>(
        static_cast<intptr_t>(nativeFrameHandle));
    std::vector<Detection> detections = detector->detectGStreamerFrame(
        holder, static_cast<float>(confidenceThreshold), static_cast<float>(nmsThreshold));
    return detector->detectionsToJava(env, detections);
  } catch (const std::exception& ex) {
    throwUnsupportedOperation(env, ex.what());
    return nullptr;
  }
}

}  // extern "C"
