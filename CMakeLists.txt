cmake_minimum_required(VERSION 3.22)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86)
endif()
project(tact_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(TACT_ENABLE_CUSPARSELT "Enable cuSPARSELt backend" OFF)
option(TACT_REFERENCE_OPT_LEVEL "Compile CUDA kernels at -O0 like historical TACT" ON)

if (TACT_REFERENCE_OPT_LEVEL)
  string(REGEX REPLACE "(^| )-O[0-9s]" "\\1" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
  string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -O0")
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(tact_bench
  src/main.cu
  src/common.cu
  src/matrix_builders.cu
  src/device_kernels.cu
  src/historical_sfu_kernels.cu
  src/historical_sp_kernels.cu
  src/baseline_engine.cu
  src/wmma_engine.cu
  src/cusparse_engine.cu
  src/cusparselt_engine.cu
  src/runner.cu
)

target_include_directories(tact_bench PRIVATE include)
target_link_libraries(tact_bench PRIVATE CUDA::cudart CUDA::cusparse)

target_compile_options(tact_bench PRIVATE
<<<<<<< HEAD
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O0>
=======
  # Do not enable --use_fast_math target-wide: baseline_sp must retain standard
  # libdevice semantics. baseline_fast explicitly calls CUDA fast intrinsics.
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
>>>>>>> e30f8f9 (Changing SFU/SP/Tensore_Core kernels according to the real TACT)
)

# The old SFU programs were built with --use_fast_math, whereas the old SP
# programs deliberately used standard libdevice sinf/cosf/tanf. Preserve that
# source-level split instead of applying fast math to the whole benchmark.
set_source_files_properties(src/historical_sfu_kernels.cu PROPERTIES
  COMPILE_OPTIONS "--use_fast_math"
)

# All device functions are defined in the same translation unit as their
# callers. Avoid relocatable device code: the historical standalone nvcc
# builds did not use -rdc=true, and RDC changes the standard-libdevice SP
# kernel's call frame and generated resource profile.
set_target_properties(tact_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if (TACT_ENABLE_CUSPARSELT)
  target_compile_definitions(tact_bench PRIVATE TACT_ENABLE_CUSPARSELT=1)
  if (DEFINED CUSPARSELT_ROOT)
    target_include_directories(tact_bench PRIVATE ${CUSPARSELT_ROOT}/include)
    target_link_directories(tact_bench PRIVATE ${CUSPARSELT_ROOT}/lib ${CUSPARSELT_ROOT}/lib64)
  elseif(DEFINED ENV{CUSPARSELT_ROOT})
    target_include_directories(tact_bench PRIVATE $ENV{CUSPARSELT_ROOT}/include)
    target_link_directories(tact_bench PRIVATE $ENV{CUSPARSELT_ROOT}/lib $ENV{CUSPARSELT_ROOT}/lib64)
  endif()
  target_link_libraries(tact_bench PRIVATE cusparseLt)
endif()

install(TARGETS tact_bench RUNTIME DESTINATION bin)
