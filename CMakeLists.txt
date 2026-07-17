cmake_minimum_required(VERSION 3.22)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
project(tact_project LANGUAGES CXX CUDA)

include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(TACT_CUDA_O0 "Compile legacy CUDA translation units at -O0 for TACT stress measurements" ON)

# CMake's Release CUDA flags normally contain -O3.  Do not combine that
# target-wide flag with per-source -O0/-O3 choices, because nvcc reports
# "incompatible redefinition for option optimize".  Keep the non-optimization
# Release flags (for example -DNDEBUG), and assign exactly one optimization
# level to each CUDA translation unit below.
string(REGEX REPLACE "(^|[ \t])-O([0-3]|s|fast)([ \t]|$)" " "
       CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
string(REGEX REPLACE "[ \t]+" " " CMAKE_CUDA_FLAGS_RELEASE
       "${CMAKE_CUDA_FLAGS_RELEASE}")
string(STRIP "${CMAKE_CUDA_FLAGS_RELEASE}" CMAKE_CUDA_FLAGS_RELEASE)

find_package(CUDAToolkit REQUIRED)

add_executable(tact_bench
  src/main.cu
  src/common.cu
  src/matrix_builders.cu
  src/device_kernels.cu
  src/sfu_kernels.cu
  src/sp_kernels.cu
  src/scalar_engine.cu
  src/wmma_engine.cu
  src/cusparse_engine.cu
  src/sparse_tc_engine.cu
  src/runner.cu
)

target_include_directories(tact_bench PRIVATE include)
target_link_libraries(tact_bench PRIVATE CUDA::cudart CUDA::cusparse)

target_compile_options(tact_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O0>
)


# Assign one and only one optimization level to every benchmark CUDA source.
# The legacy engines retain the historical -O0 stress configuration when
# TACT_CUDA_O0=ON.  The custom sparse Tensor Core translation unit is always
# compiled at -O3; its volatile mma.sp.sync loop still prevents algebraic
# collapse of stress-repeat.
set(TACT_LEGACY_CUDA_SOURCES
  src/main.cu
  src/common.cu
  src/matrix_builders.cu
  src/device_kernels.cu
  src/sfu_kernels.cu
  src/sp_kernels.cu
  src/scalar_engine.cu
  src/wmma_engine.cu
  src/cusparse_engine.cu
  src/runner.cu
)

if (TACT_CUDA_O0)
  set(TACT_LEGACY_CUDA_OPT -O0)
else()
  set(TACT_LEGACY_CUDA_OPT -O0)
endif()

set_property(SOURCE ${TACT_LEGACY_CUDA_SOURCES}
  APPEND PROPERTY COMPILE_OPTIONS ${TACT_LEGACY_CUDA_OPT})
set_property(SOURCE src/sparse_tc_engine.cu
  APPEND PROPERTY COMPILE_OPTIONS -O0)

# SFU and SP are separate translation units so fast math never leaks into SP.
set_property(SOURCE src/sfu_kernels.cu
  APPEND PROPERTY COMPILE_OPTIONS --use_fast_math)

# All device functions are defined in the same translation unit as their
# callers. Avoid relocatable device code because it changes the standard
# libdevice SP kernel's call frame and generated resource profile.
set_target_properties(tact_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

install(TARGETS tact_bench RUNTIME DESTINATION bin)

if (BUILD_TESTING)
  add_executable(tact_layout_test
    tests/matrix_layout_test.cu
    src/common.cu
    src/matrix_builders.cu
  )
  target_include_directories(tact_layout_test PRIVATE include)
  target_link_libraries(tact_layout_test PRIVATE CUDA::cudart CUDA::cusparse)
  target_compile_options(tact_layout_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  set_target_properties(tact_layout_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
  )
  add_test(NAME matrix_layout COMMAND tact_layout_test)
endif()
