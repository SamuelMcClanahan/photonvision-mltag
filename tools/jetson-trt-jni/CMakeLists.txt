cmake_minimum_required(VERSION 3.18)

project(pv_tensorrt_jni LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc)
find_package(PkgConfig REQUIRED)
pkg_check_modules(
  GSTREAMER
  REQUIRED
  IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-video-1.0)

if(NOT DEFINED JAVA_HOME)
  if(DEFINED ENV{JAVA_HOME})
    set(JAVA_HOME "$ENV{JAVA_HOME}")
  else()
    set(JAVA_HOME "/usr/lib/jvm/java-25-openjdk-arm64")
  endif()
endif()

set(JNI_INCLUDE_DIRS "${JAVA_HOME}/include" "${JAVA_HOME}/include/linux")
foreach(JNI_INCLUDE_DIR ${JNI_INCLUDE_DIRS})
  if(NOT EXISTS "${JNI_INCLUDE_DIR}")
    message(FATAL_ERROR "JNI include directory not found: ${JNI_INCLUDE_DIR}")
  endif()
endforeach()

find_path(
  TENSORRT_INCLUDE_DIR
  NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu /usr/include /usr/local/include)
find_library(
  TENSORRT_LIBRARY
  nvinfer
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)
find_library(
  TENSORRT_PLUGIN_LIBRARY
  nvinfer_plugin
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib /usr/local/lib)
find_library(
  NVBUFSURFACE_LIBRARY
  NAMES nvbufsurface
  PATHS /usr/lib/aarch64-linux-gnu/nvidia
  REQUIRED)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT headers/libraries were not found")
endif()

add_library(pv_tensorrt_jni SHARED src/TensorRTJNI.cpp src/TensorRTPreprocess.cu)
target_include_directories(
  pv_tensorrt_jni
  PRIVATE
    ${TENSORRT_INCLUDE_DIR}
    ${JNI_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    /usr/src/jetson_multimedia_api/include)

set(TENSORRT_LIBS ${TENSORRT_LIBRARY})
if(TENSORRT_PLUGIN_LIBRARY)
  list(APPEND TENSORRT_LIBS ${TENSORRT_PLUGIN_LIBRARY})
endif()

target_link_libraries(
  pv_tensorrt_jni
  PRIVATE
    CUDA::cudart
    PkgConfig::GSTREAMER
    ${TENSORRT_LIBS}
    ${NVBUFSURFACE_LIBRARY}
    ${OpenCV_LIBS})

target_compile_options(
  pv_tensorrt_jni
  PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
