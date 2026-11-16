cmake_minimum_required(VERSION 3.18)

project(pv_gstreamer_capture_jni LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(
  GSTREAMER
  REQUIRED
  IMPORTED_TARGET
  gstreamer-1.0
  gstreamer-app-1.0
  gstreamer-video-1.0)
find_package(OpenCV REQUIRED COMPONENTS core)

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

add_library(pv_gstreamer_capture_jni SHARED src/GStreamerCaptureJNI.cpp)
target_include_directories(
  pv_gstreamer_capture_jni
  PRIVATE
    ${JNI_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(
  pv_gstreamer_capture_jni
  PRIVATE
    PkgConfig::GSTREAMER
    ${OpenCV_LIBS})
target_compile_options(
  pv_gstreamer_capture_jni
  PRIVATE
    -Wall
    -Wextra
    -Wpedantic)
