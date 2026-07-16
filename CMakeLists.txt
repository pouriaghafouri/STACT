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

option(TACT_ENABLE_CUSPARSELT "Enable cuSPARSELt backend" OFF)
option(TACT_CUDA_O0 "Compile CUDA kernels at -O0 for TACT stress measurements" ON)

if (TACT_CUDA_O0)
  string(REGEX REPLACE "(^| )-O[0-9s]" "\\1" CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE}")
  string(APPEND CMAKE_CUDA_FLAGS_RELEASE " -O0")
endif()

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
  src/cusparselt_engine.cu
  src/runner.cu
)

target_include_directories(tact_bench PRIVATE include)
target_link_libraries(tact_bench PRIVATE CUDA::cudart CUDA::cusparse)

target_compile_options(tact_bench PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -lineinfo>
  $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

# SFU and SP are separate translation units so fast math never leaks into SP.
set_source_files_properties(src/sfu_kernels.cu PROPERTIES
  COMPILE_OPTIONS "--use_fast_math"
)

# All device functions are defined in the same translation unit as their
# callers. Avoid relocatable device code because it changes the standard
# libdevice SP kernel's call frame and generated resource profile.
set_target_properties(tact_bench PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

if (TACT_ENABLE_CUSPARSELT)
  set(_cusparselt_root "")
  if (DEFINED CUSPARSELT_ROOT)
    set(_cusparselt_root "${CUSPARSELT_ROOT}")
  elseif(DEFINED ENV{CUSPARSELT_ROOT})
    set(_cusparselt_root "$ENV{CUSPARSELT_ROOT}")
  endif()

  if (_cusparselt_root)
    find_path(TACT_CUSPARSELT_INCLUDE_DIR cusparseLt.h
      PATHS "${_cusparselt_root}/include" NO_DEFAULT_PATH NO_CACHE)
    find_library(TACT_CUSPARSELT_LIBRARY cusparseLt
      PATHS "${_cusparselt_root}/lib" "${_cusparselt_root}/lib64"
      NO_DEFAULT_PATH NO_CACHE)
  else()
    find_path(TACT_CUSPARSELT_INCLUDE_DIR cusparseLt.h NO_CACHE)
    find_library(TACT_CUSPARSELT_LIBRARY cusparseLt NO_CACHE)
  endif()

  if (NOT TACT_CUSPARSELT_INCLUDE_DIR OR NOT TACT_CUSPARSELT_LIBRARY)
    message(FATAL_ERROR
      "cuSPARSELt header/library pair not found. Set CUSPARSELT_ROOT to one "
      "cuSPARSELt archive, then reconfigure build-slt.")
  endif()

  file(STRINGS "${TACT_CUSPARSELT_INCLUDE_DIR}/cusparseLt.h"
    _cusparselt_version_lines
    REGEX "^#define CUSPARSELT_VER_(MAJOR|MINOR) [0-9]+$")
  string(REGEX MATCH "CUSPARSELT_VER_MAJOR +([0-9]+)"
    _major_match "${_cusparselt_version_lines}")
  set(_header_major "${CMAKE_MATCH_1}")
  string(REGEX MATCH "CUSPARSELT_VER_MINOR +([0-9]+)"
    _minor_match "${_cusparselt_version_lines}")
  set(_header_minor "${CMAKE_MATCH_1}")
  get_filename_component(_cusparselt_library_real
    "${TACT_CUSPARSELT_LIBRARY}" REALPATH)
  get_filename_component(_cusparselt_library_name
    "${_cusparselt_library_real}" NAME)
  if (_cusparselt_library_name MATCHES
      "libcusparseLt\\.so\\.([0-9]+)\\.([0-9]+)")
    set(_library_major "${CMAKE_MATCH_1}")
    set(_library_minor "${CMAKE_MATCH_2}")
    if (NOT _header_major STREQUAL _library_major OR
        NOT _header_minor STREQUAL _library_minor)
      message(FATAL_ERROR
        "cuSPARSELt ABI mismatch: header is ${_header_major}.${_header_minor} "
        "(${TACT_CUSPARSELT_INCLUDE_DIR}/cusparseLt.h), but library is "
        "${_library_major}.${_library_minor} (${_cusparselt_library_real}). "
        "Set CUSPARSELT_ROOT to an archive containing a matching include/ "
        "and lib/ pair. Mixing these versions produces invalid handles at "
        "cusparseLtSpMMACompress().")
    endif()
  endif()
  if (CUDAToolkit_VERSION_MAJOR GREATER_EQUAL 12 AND
      _header_major STREQUAL "0" AND _header_minor LESS 4)
    message(FATAL_ERROR
      "cuSPARSELt ${_header_major}.${_header_minor} is linked against the "
      "CUDA 11 runtime, but this build uses CUDA ${CUDAToolkit_VERSION}. "
      "Use a matching cuSPARSELt 0.4+ archive for CUDA 12.")
  endif()

  target_compile_definitions(tact_bench PRIVATE TACT_ENABLE_CUSPARSELT=1)
  target_include_directories(tact_bench PRIVATE ${TACT_CUSPARSELT_INCLUDE_DIR})
  target_link_libraries(tact_bench PRIVATE ${TACT_CUSPARSELT_LIBRARY})
  get_filename_component(_cusparselt_library_dir
    "${TACT_CUSPARSELT_LIBRARY}" DIRECTORY)
  set_property(TARGET tact_bench APPEND PROPERTY BUILD_RPATH
    "${_cusparselt_library_dir}")
endif()

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
