cmake_minimum_required(VERSION 3.18)
project(bnudc_trt_infer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS
        ${TENSORRT_ROOT}/include
        /usr/include/aarch64-linux-gnu
        /usr/include/x86_64-linux-gnu
        /usr/include)
find_library(TENSORRT_LIBRARY nvinfer
    HINTS
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib64
        ${TENSORRT_ROOT}/targets/aarch64-linux-gnu/lib
        ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu)
find_library(TENSORRT_PLUGIN_LIBRARY nvinfer_plugin
    HINTS
        ${TENSORRT_ROOT}/lib
        ${TENSORRT_ROOT}/lib64
        ${TENSORRT_ROOT}/targets/aarch64-linux-gnu/lib
        ${TENSORRT_ROOT}/targets/x86_64-linux-gnu/lib
        /usr/lib/aarch64-linux-gnu
        /usr/lib/x86_64-linux-gnu)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "TensorRT headers/library not found. Set TENSORRT_ROOT or CMAKE_PREFIX_PATH.")
endif()

add_executable(bnudc_trt_infer src/main.cpp)
set_target_properties(bnudc_trt_infer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
target_include_directories(bnudc_trt_infer PRIVATE ${TENSORRT_INCLUDE_DIR})
target_link_libraries(bnudc_trt_infer PRIVATE CUDA::cudart ${TENSORRT_LIBRARY})
if(TENSORRT_PLUGIN_LIBRARY)
    target_link_libraries(bnudc_trt_infer PRIVATE ${TENSORRT_PLUGIN_LIBRARY})
endif()
target_compile_options(bnudc_trt_infer PRIVATE -Wall -Wextra -Wpedantic)

include(GNUInstallDirs)
install(TARGETS bnudc_trt_infer RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
