#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

void checkCuda(cudaError_t status, const char* what) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
  }
}

__device__ float pixelChannel(
    const uint8_t* src, int width, int height, int channels, size_t stepBytes, int x, int y,
    int rgbChannel, int channelOrder) {
  x = max(0, min(width - 1, x));
  y = max(0, min(height - 1, y));
  const uint8_t* pixel = src + static_cast<size_t>(y) * stepBytes + static_cast<size_t>(x) * channels;
  if (channels == 1) {
    return static_cast<float>(pixel[0]);
  }

  const bool rgbOrder = channelOrder == 1;
  int sourceChannel = rgbOrder ? rgbChannel : 2 - rgbChannel;
  return static_cast<float>(pixel[sourceChannel]);
}

__device__ float sampleBilinear(
    const uint8_t* src, int width, int height, int channels, size_t stepBytes, float srcX,
    float srcY, int rgbChannel, int channelOrder) {
  if (srcX < 0.0F || srcY < 0.0F || srcX > static_cast<float>(width - 1) ||
      srcY > static_cast<float>(height - 1)) {
    return 114.0F / 255.0F;
  }

  const int x0 = static_cast<int>(floorf(srcX));
  const int y0 = static_cast<int>(floorf(srcY));
  const int x1 = min(width - 1, x0 + 1);
  const int y1 = min(height - 1, y0 + 1);
  const float wx = srcX - static_cast<float>(x0);
  const float wy = srcY - static_cast<float>(y0);

  const float p00 = pixelChannel(src, width, height, channels, stepBytes, x0, y0, rgbChannel, channelOrder);
  const float p10 = pixelChannel(src, width, height, channels, stepBytes, x1, y0, rgbChannel, channelOrder);
  const float p01 = pixelChannel(src, width, height, channels, stepBytes, x0, y1, rgbChannel, channelOrder);
  const float p11 = pixelChannel(src, width, height, channels, stepBytes, x1, y1, rgbChannel, channelOrder);

  const float top = p00 + (p10 - p00) * wx;
  const float bottom = p01 + (p11 - p01) * wx;
  return (top + (bottom - top) * wy) / 255.0F;
}

template <typename T>
__global__ void preprocessKernel(
    const uint8_t* src, int srcWidth, int srcHeight, int channels, size_t srcStepBytes, T* dst,
    int dstWidth, int dstHeight, float scale, float dx, float dy, int channelOrder) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= dstWidth || y >= dstHeight) {
    return;
  }

  // Match OpenCV resize's pixel-center mapping. This keeps the CUDA letterbox path aligned with
  // the CPU/OpenCV letterbox path that the sidecar detector uses.
  const float srcX = (static_cast<float>(x) - dx + 0.5F) / scale - 0.5F;
  const float srcY = (static_cast<float>(y) - dy + 0.5F) / scale - 0.5F;
  const size_t pixelIndex = static_cast<size_t>(y) * dstWidth + x;
  const size_t plane = static_cast<size_t>(dstWidth) * dstHeight;

  const float red = sampleBilinear(src, srcWidth, srcHeight, channels, srcStepBytes, srcX, srcY, 0, channelOrder);
  const float green = sampleBilinear(src, srcWidth, srcHeight, channels, srcStepBytes, srcX, srcY, 1, channelOrder);
  const float blue = sampleBilinear(src, srcWidth, srcHeight, channels, srcStepBytes, srcX, srcY, 2, channelOrder);

  if constexpr (std::is_same_v<T, __half>) {
    dst[pixelIndex] = __float2half(red);
    dst[plane + pixelIndex] = __float2half(green);
    dst[2 * plane + pixelIndex] = __float2half(blue);
  } else {
    dst[pixelIndex] = red;
    dst[plane + pixelIndex] = green;
    dst[2 * plane + pixelIndex] = blue;
  }
}

}  // namespace

void launchTensorRTPreprocess(
    const uint8_t* srcDevice, int srcWidth, int srcHeight, int channels, size_t srcStepBytes,
    void* dstDevice, int dstWidth, int dstHeight, float scale, float dx, float dy, bool fp16,
    int channelOrder, cudaStream_t stream) {
  const dim3 block(16, 16);
  const dim3 grid((dstWidth + block.x - 1) / block.x, (dstHeight + block.y - 1) / block.y);
  if (fp16) {
    preprocessKernel<<<grid, block, 0, stream>>>(
        srcDevice, srcWidth, srcHeight, channels, srcStepBytes, static_cast<__half*>(dstDevice),
        dstWidth, dstHeight, scale, dx, dy, channelOrder);
  } else {
    preprocessKernel<<<grid, block, 0, stream>>>(
        srcDevice, srcWidth, srcHeight, channels, srcStepBytes, static_cast<float*>(dstDevice),
        dstWidth, dstHeight, scale, dx, dy, channelOrder);
  }
  checkCuda(cudaGetLastError(), "preprocessKernel");
}
