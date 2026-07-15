cmake_minimum_required(VERSION 3.22)
project(tact_project LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(TACT_ENABLE_CUSPARSELT "Enable cuSPARSELt backend" OFF)

find_package(CUDAToolkit REQUIRED)

add_executable(tact_bench
  src/main.cu
  src/common.cu
  src/matrix_builders.cu
  src/device_kernels.cu
  src/baseline_engine.cu
  src/wmma_engine.cu
  src/cusparse_engine.cu
  src/cusparselt_engine.cu
  src/runner.cu
)

target_include_directories(tact_bench PRIVATE include)
target_link_libraries(tact_bench PRIVATE CUDA::cudart CUDA::cusparse)

target_compile_options(tact_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --use_fast_math -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O0>
)

set_target_properties(tact_bench PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

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
