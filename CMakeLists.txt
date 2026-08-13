cmake_minimum_required(VERSION 3.24...4.4)

if(DEFINED SKBUILD_PROJECT_VERSION)
  project(tonari VERSION "${SKBUILD_PROJECT_VERSION}" LANGUAGES CXX)
else()
  project(tonari VERSION 0.1.0 LANGUAGES CXX)
endif()

find_package(Threads REQUIRED)

add_library(tonari_geometry STATIC csrc/core/geometry.cpp)
add_library(tonari::geometry ALIAS tonari_geometry)
target_compile_features(tonari_geometry PUBLIC cxx_std_20)
target_include_directories(
  tonari_geometry PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc/core>)
set_target_properties(tonari_geometry PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(
  tonari_cpu STATIC csrc/core/neighbors_cpu.cpp csrc/core/thread_pool.cpp)
add_library(tonari::cpu ALIAS tonari_cpu)
target_compile_features(tonari_cpu PUBLIC cxx_std_20)
target_include_directories(
  tonari_cpu PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/csrc/core>)
target_link_libraries(tonari_cpu PUBLIC tonari::geometry Threads::Threads)
set_target_properties(tonari_cpu PROPERTIES POSITION_INDEPENDENT_CODE ON)

option(
  TONARI_BUILD_PYTHON
  "Build the Tonari Python extension modules"
  ${PROJECT_IS_TOP_LEVEL})

if(TONARI_BUILD_PYTHON)
  if(PROJECT_IS_TOP_LEVEL AND DEFINED ENV{BUILD_CUDA})
    if(NOT "$ENV{BUILD_CUDA}" MATCHES "^(0|1)$")
      message(FATAL_ERROR "BUILD_CUDA must be '0' or '1'")
    endif()
    if(NOT DEFINED TONARI_BUILD_CUDA)
      set(_tonari_build_cuda_default "$ENV{BUILD_CUDA}")
    endif()
  endif()
  if(NOT DEFINED _tonari_build_cuda_default)
    set(_tonari_build_cuda_default ON)
  endif()
  option(
    TONARI_BUILD_CUDA
    "Build the Tonari Torch CUDA extension module"
    ${_tonari_build_cuda_default})

  find_package(
    Python 3.11 REQUIRED
    COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)

  execute_process(
    COMMAND
      "${Python_EXECUTABLE}" -c
      "import json, pathlib, torch; print(json.dumps({'version': torch.__version__, 'cuda': torch.version.cuda or '', 'root': str(pathlib.Path(torch.__file__).parent), 'cxx11_abi': int(torch.compiled_with_cxx11_abi())}))"
    RESULT_VARIABLE _tonari_torch_query_status
    OUTPUT_VARIABLE _tonari_torch_query_output
    ERROR_VARIABLE _tonari_torch_query_error
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(NOT _tonari_torch_query_status EQUAL 0)
    message(
      FATAL_ERROR
        "Tonari requires PyTorch in the build environment:\n${_tonari_torch_query_error}")
  endif()
  string(JSON TONARI_TORCH_VERSION GET "${_tonari_torch_query_output}" version)
  string(JSON TONARI_TORCH_CUDA_VERSION GET "${_tonari_torch_query_output}" cuda)
  string(JSON TONARI_TORCH_ROOT GET "${_tonari_torch_query_output}" root)
  string(JSON TONARI_TORCH_CXX11_ABI GET "${_tonari_torch_query_output}" cxx11_abi)

  if(TONARI_BUILD_CUDA)
    if(TONARI_TORCH_CUDA_VERSION STREQUAL "")
      message(
        FATAL_ERROR
          "The default CUDA build requires a CUDA-enabled PyTorch installation; set BUILD_CUDA=0 for a CPU-only build")
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
      set(_tonari_use_native_cuda_architectures ON)
    endif()
    include(CheckLanguage)
    check_language(CUDA)
    if(NOT CMAKE_CUDA_COMPILER)
      message(
        FATAL_ERROR
          "The default CUDA build requires a CUDA toolkit containing nvcc; set BUILD_CUDA=0 for a CPU-only build")
    endif()
    enable_language(CUDA)
    if(_tonari_use_native_cuda_architectures)
      set(CMAKE_CUDA_ARCHITECTURES native)
    endif()
    find_package(CUDAToolkit REQUIRED)
    string(REGEX MATCH "^[0-9]+" _tonari_torch_cuda_major "${TONARI_TORCH_CUDA_VERSION}")
    string(REGEX MATCH "^[0-9]+" _tonari_toolkit_cuda_major "${CUDAToolkit_VERSION}")
    if(NOT "${_tonari_torch_cuda_major}" STREQUAL "${_tonari_toolkit_cuda_major}")
      message(
        FATAL_ERROR
          "PyTorch uses CUDA ${TONARI_TORCH_CUDA_VERSION}, but the local toolkit is ${CUDAToolkit_VERSION}; their major versions must match")
    endif()
  endif()

  set(_tonari_torch_libraries c10 torch torch_cpu torch_python)
  if(TONARI_BUILD_CUDA)
    list(APPEND _tonari_torch_libraries c10_cuda torch_cuda)
  endif()
  set(TONARI_TORCH_LIBRARIES)
  foreach(_tonari_torch_library IN LISTS _tonari_torch_libraries)
    string(TOUPPER "${_tonari_torch_library}" _tonari_torch_library_upper)
    find_library(
      TONARI_${_tonari_torch_library_upper}_LIBRARY
      NAMES "${_tonari_torch_library}"
      PATHS "${TONARI_TORCH_ROOT}/lib"
      NO_DEFAULT_PATH NO_CACHE REQUIRED)
    list(
      APPEND TONARI_TORCH_LIBRARIES
      "${TONARI_${_tonari_torch_library_upper}_LIBRARY}")
  endforeach()

  add_library(tonari_torch INTERFACE)
  target_include_directories(
    tonari_torch SYSTEM INTERFACE
    "${TONARI_TORCH_ROOT}/include"
    "${TONARI_TORCH_ROOT}/include/torch/csrc/api/include")
  target_compile_definitions(
    tonari_torch
    INTERFACE _GLIBCXX_USE_CXX11_ABI=${TONARI_TORCH_CXX11_ABI})
  target_link_libraries(tonari_torch INTERFACE ${TONARI_TORCH_LIBRARIES})

  function(tonari_configure_torch_module target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_compile_definitions(
      ${target}
      PRIVATE TORCH_API_INCLUDE_EXTENSION_H TORCH_EXTENSION_NAME=${target})
    target_link_libraries(${target} PRIVATE tonari_torch)
    set_target_properties(
      ${target}
      PROPERTIES CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
  endfunction()

  pybind11_add_module(_numpy_cpu MODULE csrc/numpy/bindings.cpp)
  target_link_libraries(_numpy_cpu PRIVATE tonari::cpu)

  Python_add_library(
    _torch_cpu MODULE WITH_SOABI csrc/torch/bindings_cpu.cpp)
  tonari_configure_torch_module(_torch_cpu)
  target_link_libraries(_torch_cpu PRIVATE tonari::cpu)

  set(_tonari_python_targets _numpy_cpu _torch_cpu)
  if(TONARI_BUILD_CUDA)
    Python_add_library(
      _torch_cuda MODULE WITH_SOABI
      csrc/torch/bindings_cuda.cpp
      csrc/torch/neighbors_cuda.cu
      csrc/torch/neighbors_cell_cuda.cu)
    tonari_configure_torch_module(_torch_cuda)
    target_compile_options(
      _torch_cuda
      PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-extended-lambda>
        $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
        $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    target_link_libraries(
      _torch_cuda PRIVATE tonari::geometry CUDA::cudart)
    list(APPEND _tonari_python_targets _torch_cuda)
  endif()

  if(TONARI_BUILD_CUDA)
    set(TONARI_PY_BUILD_WITH_CUDA True)
    set(TONARI_PY_CUDA_TOOLKIT_VERSION "\"${CUDAToolkit_VERSION}\"")
  else()
    set(TONARI_PY_BUILD_WITH_CUDA False)
    set(TONARI_PY_CUDA_TOOLKIT_VERSION None)
  endif()
  if(TONARI_TORCH_CUDA_VERSION STREQUAL "")
    set(TONARI_PY_TORCH_CUDA_VERSION None)
  else()
    set(TONARI_PY_TORCH_CUDA_VERSION "\"${TONARI_TORCH_CUDA_VERSION}\"")
  endif()
  configure_file(
    cmake/_build_info.py.in generated/tonari/_build_info.py @ONLY)

  install(
    TARGETS ${_tonari_python_targets}
    LIBRARY DESTINATION tonari
    RUNTIME DESTINATION tonari)
  install(
    FILES "${CMAKE_CURRENT_BINARY_DIR}/generated/tonari/_build_info.py"
    DESTINATION tonari)

  message(
    STATUS
      "Tonari providers: NumPy CPU, Torch CPU, CUDA=${TONARI_BUILD_CUDA}; Torch=${TONARI_TORCH_VERSION}; Torch CUDA=${TONARI_TORCH_CUDA_VERSION}; toolkit=${CUDAToolkit_VERSION}")
endif()
