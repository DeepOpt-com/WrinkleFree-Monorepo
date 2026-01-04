# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# examples - BitNet/DLM essentials only
# Other examples moved to _deprecated/examples/

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

if (EMSCRIPTEN)
else()
    add_subdirectory(main)
    add_subdirectory(quantize)
endif()
