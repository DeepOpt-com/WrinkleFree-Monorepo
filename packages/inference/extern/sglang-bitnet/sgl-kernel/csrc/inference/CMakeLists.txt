cmake_minimum_required(VERSION 3.18)
project(sgl_kernel_inference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SIMD flags (same as bitnet library)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    message(STATUS "Inference: Enabling AVX2/FMA for x86_64")

    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512bw -mavx512vl")
        message(STATUS "Inference: Enabling AVX512 support")
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    message(STATUS "Inference: Enabling NEON for ARM64")

    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("
        #include <arm_neon.h>
        int main() {
            int8x16_t a = vdupq_n_s8(1);
            int8x16_t b = vdupq_n_s8(1);
            int32x4_t c = vdupq_n_s32(0);
            c = vdotq_s32(c, a, b);
            return 0;
        }
    " COMPILER_SUPPORTS_DOTPROD)
    if(COMPILER_SUPPORTS_DOTPROD)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.2-a+dotprod")
        message(STATUS "Inference: Enabling ARM dot product extension")
    endif()
else()
    message(FATAL_ERROR "Inference: Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# OpenMP for multi-threading
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "Inference: OpenMP enabled")
endif()

# Source files
set(INFERENCE_SOURCES
    bitnet_engine.cpp
    kv_cache.cpp
)

# Build as both static and shared library
add_library(sgl_kernel_inference_static STATIC ${INFERENCE_SOURCES})
add_library(sgl_kernel_inference SHARED ${INFERENCE_SOURCES})

# Link to BitNet SIMD kernels
target_link_libraries(sgl_kernel_inference_static PRIVATE sgl_kernel_bitnet)
target_link_libraries(sgl_kernel_inference PRIVATE sgl_kernel_bitnet)

# Include paths
target_include_directories(sgl_kernel_inference_static PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../bitnet
)
target_include_directories(sgl_kernel_inference PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../bitnet
)

# Make sure the bitnet library is built first
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../bitnet ${CMAKE_CURRENT_BINARY_DIR}/bitnet)

# Set shared library properties
set_target_properties(sgl_kernel_inference PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
    OUTPUT_NAME "sgl_kernel_inference"
)

# Install
install(TARGETS sgl_kernel_inference sgl_kernel_inference_static
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES bitnet_engine.h kv_cache.h DESTINATION include/sgl_kernel/inference)
