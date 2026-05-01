cmake_minimum_required(VERSION 3.18)
project(chess LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Build options ─────────────────────────────────────────────────────────
option(USE_OMP  "Enable OpenMP root-parallel search" ON)
option(USE_CUDA "Enable CUDA GPU batch evaluation"   OFF)

# ── Common sources ─────────────────────────────────────────────────────────
set(COMMON_SRC
    src/board.cpp
    src/movegen.cpp
    src/eval.cpp
    src/search_serial.cpp
    src/search_omp.cpp
    src/search_cuda.cpp
    src/main.cpp
)

# ── CUDA sources ──────────────────────────────────────────────────────────
if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND COMMON_SRC cuda/eval_cuda.cu)
endif()

# ── Executable ────────────────────────────────────────────────────────────
add_executable(chess ${COMMON_SRC})

target_include_directories(chess PRIVATE src cuda)

# ── Compiler flags ─────────────────────────────────────────────────────────
target_compile_options(chess PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -march=native -Wall -Wextra -Wno-unused-parameter>
)

if(USE_CUDA)
    target_compile_options(chess PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
    set_target_properties(chess PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_compile_definitions(chess PRIVATE USE_CUDA)
endif()

# ── OpenMP ────────────────────────────────────────────────────────────────
if(USE_OMP)
    # Apple Clang needs libomp from Homebrew; hint CMake to find it
    if(APPLE)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        find_library(OpenMP_omp_LIBRARY omp HINTS /opt/homebrew/opt/libomp/lib)
        set(OpenMP_CXX_INCLUDE_DIRS "/opt/homebrew/opt/libomp/include")
    endif()
    find_package(OpenMP REQUIRED)
    target_link_libraries(chess PRIVATE OpenMP::OpenMP_CXX)
    if(APPLE)
        target_include_directories(chess PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    endif()
    target_compile_definitions(chess PRIVATE USE_OMP)
endif()

# ── Install / info ─────────────────────────────────────────────────────────
message(STATUS "Build options:")
message(STATUS "  OpenMP : ${USE_OMP}")
message(STATUS "  CUDA   : ${USE_CUDA}")
message(STATUS "")
message(STATUS "Run targets:")
message(STATUS "  bench (default)   ./chess bench --depth 6")
message(STATUS "  perft             ./chess perft --depth 5")
message(STATUS "  uci               ./chess uci")
