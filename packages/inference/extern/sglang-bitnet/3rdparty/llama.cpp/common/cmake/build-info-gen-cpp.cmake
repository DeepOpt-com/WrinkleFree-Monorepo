# Build info generator - placeholder script
set(BUILD_NUMBER 1)
set(BUILD_COMMIT "unknown")
set(BUILD_COMPILER "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
set(BUILD_TARGET "x86_64")

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/common/build-info.cpp.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/build-info.cpp"
    @ONLY
)
