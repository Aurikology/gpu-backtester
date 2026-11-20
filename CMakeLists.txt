cmake_minimum_required(VERSION 3.25)
project(gpu_backtester LANGUAGES CXX)

# GBT_GPU selects the GPU backend:
#   OFF     CPU only (CI runners, machines without a GPU)
#   NVIDIA  HIP source compiled through nvcc (HIP_PLATFORM=nvidia)
#   AMD     HIP source compiled with ROCm's clang (HIP_PLATFORM=amd)
set(GBT_GPU "NVIDIA" CACHE STRING "GPU backend: OFF, NVIDIA or AMD")
set_property(CACHE GBT_GPU PROPERTY STRINGS OFF NVIDIA AMD)
set(ROCM_PATH "/opt/rocm" CACHE PATH "ROCm install (HIP headers, hipCUB)")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# No fused multiply-add contraction anywhere: CPU and GPU must produce bit-identical
# floats so the backends can be checked against each other exactly.
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>)

set(CORE_SOURCES src/data.cpp src/strategy.cpp src/engine.cpp src/montecarlo.cpp src/profiler.cpp)
set(GPU_SOURCES src/gpu_engine.cu src/prims.cu)

if(GBT_GPU STREQUAL "OFF")
  add_library(gbt_core STATIC ${CORE_SOURCES} src/gpu_stub.cpp)
elseif(GBT_GPU STREQUAL "NVIDIA")
  enable_language(CUDA)
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  set(CMAKE_CUDA_STANDARD 20)
  find_package(CUDAToolkit REQUIRED)
  add_library(gbt_core STATIC ${CORE_SOURCES} ${GPU_SOURCES})
  target_include_directories(gbt_core SYSTEM PUBLIC ${ROCM_PATH}/include)
  target_compile_definitions(gbt_core PUBLIC __HIP_PLATFORM_NVIDIA__)
  target_compile_options(gbt_core PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false
                         --expt-relaxed-constexpr -Xcompiler=-ffp-contract=off -lineinfo>)
  target_link_libraries(gbt_core PUBLIC CUDA::cudart)
elseif(GBT_GPU STREQUAL "AMD")
  enable_language(HIP)
  find_package(hip REQUIRED)
  find_package(hipcub REQUIRED)
  set_source_files_properties(${GPU_SOURCES} PROPERTIES LANGUAGE HIP)
  add_library(gbt_core STATIC ${CORE_SOURCES} ${GPU_SOURCES})
  target_compile_options(gbt_core PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-ffp-contract=off>)
  target_link_libraries(gbt_core PUBLIC hip::host hip::hipcub)
else()
  message(FATAL_ERROR "GBT_GPU must be OFF, NVIDIA or AMD")
endif()

target_include_directories(gbt_core PUBLIC include)
target_compile_options(gbt_core PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3>)

add_executable(gbt apps/gbt.cpp)
target_link_libraries(gbt PRIVATE gbt_core)

# ---- tests ----
option(GBT_TESTS "Build tests" ON)
if(GBT_TESTS)
  find_package(GTest REQUIRED)
  enable_testing()
  add_executable(gbt_tests tests/test_data.cpp tests/test_strategy.cpp tests/test_engine.cpp
                           tests/test_montecarlo.cpp)
  if(NOT GBT_GPU STREQUAL "OFF")
    target_sources(gbt_tests PRIVATE tests/test_prims.cu)
    if(GBT_GPU STREQUAL "AMD")
      set_source_files_properties(tests/test_prims.cu PROPERTIES LANGUAGE HIP)
    endif()
  endif()
  target_link_libraries(gbt_tests PRIVATE gbt_core GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(gbt_tests)
endif()
