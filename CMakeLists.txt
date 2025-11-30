cmake_minimum_required(VERSION 3.27)
project(test-optix9 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(MSVC)
  add_compile_definitions(NOMINMAX)
  add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR=1)
  add_compile_options("/utf-8")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --forward-unknown-to-host-compiler --compiler-options /utf-8")
endif()

# OpenUSD内部の警告を抑制
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN _SILENCE_CXX20_OLD_SHARED_PTR_ATOMIC_SUPPORT_DEPRECATION_WARNING)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(compile_ptx)

find_package(OpenGL REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_package(OptiX REQUIRED)
find_package(Stb REQUIRED)
find_package(CLI11 CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)

set(USD_ROOT "${PXR_ROOT}")
list(APPEND CMAKE_PREFIX_PATH "${PXR_ROOT}")
find_package(pxr REQUIRED)

string(REPLACE "." "_" PXR_VERSION_MACRO "${PXR_VERSION_STRING}")
message(STATUS "Detected USD version: ${PXR_VERSION_STRING} (macro: PXR_VERSION_${PXR_VERSION_MACRO})")
add_compile_definitions("PXR_VERSION_${PXR_VERSION_MACRO}")

add_subdirectory(external/openqmc)
add_subdirectory(oxu)
add_subdirectory(hdKanahebi)

if (${BUILD_TYPE} STREQUAL "cli")
  add_subdirectory(kanahebi-cli)
endif()
