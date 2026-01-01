cmake_minimum_required(VERSION 3.18)
project(sgl_kernel_bitnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SIMD flags based on platform
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    message(STATUS "BitNet: Enabling AVX2/FMA for x86_64")

    # Optional AVX512 support
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512)
    if(COMPILER_SUPPORTS_AVX512)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512bw -mavx512vl")
        message(STATUS "BitNet: Enabling AVX512 support")

        # Check for AVX-512 VNNI (fast int8 dot products)
        check_cxx_compiler_flag("-mavx512vnni" COMPILER_SUPPORTS_AVX512_VNNI)
        if(COMPILER_SUPPORTS_AVX512_VNNI)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512vnni")
            message(STATUS "BitNet: Enabling AVX512-VNNI support (fast int8)")
        endif()

        # Check for AVX-512 BF16
        check_cxx_compiler_flag("-mavx512bf16" COMPILER_SUPPORTS_AVX512_BF16)
        if(COMPILER_SUPPORTS_AVX512_BF16)
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512bf16")
            message(STATUS "BitNet: Enabling AVX512-BF16 support")
        endif()
    endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8-a+simd")
    message(STATUS "BitNet: Enabling NEON for ARM64")

    # Check for dot product extension
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
        message(STATUS "BitNet: Enabling ARM dot product extension")
    endif()
else()
    message(FATAL_ERROR "BitNet: Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. Only x86_64 and ARM64 are supported.")
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")

# OpenMP for multi-threading
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "BitNet: OpenMP enabled")
endif()

# Source files
set(BITNET_SOURCES
    bitnet_gemv.cpp
)

# Build as static library with PIC for shared library linking
add_library(sgl_kernel_bitnet STATIC ${BITNET_SOURCES})
set_property(TARGET sgl_kernel_bitnet PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(sgl_kernel_bitnet PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Install
install(TARGETS sgl_kernel_bitnet
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(FILES bitnet_gemv.h DESTINATION include/sgl_kernel/bitnet)
