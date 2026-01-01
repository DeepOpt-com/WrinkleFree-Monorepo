cmake_minimum_required(VERSION 3.22)
project(llmcore)

# Path to shared inference code (no duplication!)
set(INFERENCE_ROOT ${CMAKE_SOURCE_DIR}/../../../../../../inference/extern/sglang-bitnet)
set(LLAMA_CPP_ROOT ${INFERENCE_ROOT}/3rdparty/llama.cpp)
set(SGL_KERNEL_ROOT ${INFERENCE_ROOT}/sgl-kernel)
set(BITNET_DIR ${CMAKE_SOURCE_DIR}/../../../../../../inference/extern/BitNet)

# ARM optimizations + TL1 kernels (NOT TL2 - that's for x86 AVX512)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.4a+dotprod -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=armv8.4a+dotprod -O3")
add_compile_definitions(GGML_BITNET_ARM_TL1)

# Disable features not needed on Android
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)

# Include llama.cpp as subdirectory (EXCLUDE_FROM_ALL to skip tests/examples)
add_subdirectory(${LLAMA_CPP_ROOT} llama.cpp EXCLUDE_FROM_ALL)

# JNI wrapper that calls llama.cpp
add_library(llmcore SHARED
    llmcore_jni.cpp
)

target_include_directories(llmcore PRIVATE
    ${LLAMA_CPP_ROOT}/include
    ${LLAMA_CPP_ROOT}/ggml/include
)

target_link_libraries(llmcore
    llama
    ggml
    log
)
