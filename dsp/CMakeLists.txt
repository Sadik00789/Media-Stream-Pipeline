cmake_minimum_required(VERSION 3.16)
project(dsp_core C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Module path for FindSIMD
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../cmake")
include(FindSIMD)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -Wall -Wextra -pedantic ${SIMD_C_FLAGS}")
add_definitions(${SIMD_DEFINITIONS})

include_directories(
    include
    ${CMAKE_CURRENT_SOURCE_DIR}/../include
)

set(DSP_SOURCES
    src/dsp_core.c
    src/windowing.c
    src/simd_vad.c
    src/simd_fft_avx2.c
    src/simd_fft_avx512.c
    src/simd_fft_neon.c
)

# Compile static library
add_library(dsp_core STATIC ${DSP_SOURCES})
target_link_libraries(dsp_core PUBLIC m)

# Unit Tests
add_executable(test_fft tests/test_fft.c)
target_link_libraries(test_fft PRIVATE dsp_core m)

add_executable(test_vad tests/test_vad.c)
target_link_libraries(test_vad PRIVATE dsp_core m)

enable_testing()
add_test(NAME test_fft COMMAND test_fft)
add_test(NAME test_vad COMMAND test_vad)
