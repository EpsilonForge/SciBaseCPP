cmake_minimum_required(VERSION 3.20)
project(SciBaseCPP LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ============================================================================
# SciBaseCPP: A boilerplate CMake project for Scientific C++ Applications
# ============================================================================
#
# This project uses FetchContent to pull in high-quality scientific computing
# libraries directly from GitHub, so you don't have to install anything
# manually. Just configure and build.
#
# Libraries included:
#   - Kokkos mdspan    (C++23 std::mdspan reference implementation)
#   - Eigen            (Dense & sparse linear algebra)
#   - PocketFFT        (Fast Fourier Transforms, BSD-licensed)
#   - SUNDIALS/CVODE   (ODE integration, stiff & nonstiff)
#   - NLopt            (Nonlinear optimization)
#   - {fmt}            (Modern formatting)
#   - Lean-VTK         (Simple VTK file output for visualization)
#   - Catch2           (Testing framework)
# ============================================================================

include(FetchContent)

# ---------------------------------------------------------------------------
# 1. mdspan — reference implementation of C++23 std::mdspan
#    https://github.com/kokkos/mdspan
# ---------------------------------------------------------------------------
FetchContent_Declare(
    mdspan
    GIT_REPOSITORY https://github.com/kokkos/mdspan.git
    GIT_TAG        stable
)
FetchContent_MakeAvailable(mdspan)

# ---------------------------------------------------------------------------
# 2. Eigen — Dense and sparse linear algebra
#    https://gitlab.com/libeigen/eigen
# ---------------------------------------------------------------------------
FetchContent_Declare(
    Eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG        3.4.0
    GIT_SHALLOW    TRUE
)
set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
set(EIGEN_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(Eigen)

# ---------------------------------------------------------------------------
# 3. {fmt} — Modern C++ formatting (std::format backport)
#    https://github.com/fmtlib/fmt
# ---------------------------------------------------------------------------
FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG        11.1.4
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(fmt)

# ---------------------------------------------------------------------------
# 4. PocketFFT — BSD-licensed FFT library (same engine as NumPy)
#    https://github.com/mreineck/pocketfft
#    Header-only, so we just need to expose the include path.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    pocketfft
    GIT_REPOSITORY https://github.com/mreineck/pocketfft.git
    GIT_TAG        cpp
)
FetchContent_GetProperties(pocketfft)
if(NOT pocketfft_POPULATED)
    FetchContent_Populate(pocketfft)
    add_library(pocketfft INTERFACE)
    target_include_directories(pocketfft INTERFACE ${pocketfft_SOURCE_DIR})
endif()

# ---------------------------------------------------------------------------
# 5. SUNDIALS — ODE/DAE solvers from LLNL (CVODE, ARKODE, etc.)
#    https://github.com/LLNL/sundials
#    We only enable CVODE to keep the build lean.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    sundials
    GIT_REPOSITORY https://github.com/LLNL/sundials.git
    GIT_TAG        v7.2.0
    GIT_SHALLOW    TRUE
)
set(BUILD_ARKODE OFF CACHE BOOL "" FORCE)
set(BUILD_CVODE ON CACHE BOOL "" FORCE)
set(BUILD_CVODES OFF CACHE BOOL "" FORCE)
set(BUILD_IDA OFF CACHE BOOL "" FORCE)
set(BUILD_IDAS OFF CACHE BOOL "" FORCE)
set(BUILD_KINSOL OFF CACHE BOOL "" FORCE)
set(EXAMPLES_ENABLE_C OFF CACHE BOOL "" FORCE)
set(EXAMPLES_ENABLE_CXX OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(sundials)

# ---------------------------------------------------------------------------
# 6. NLopt — Nonlinear optimization library
#    https://github.com/stevengj/nlopt
# ---------------------------------------------------------------------------
FetchContent_Declare(
    nlopt
    GIT_REPOSITORY https://github.com/stevengj/nlopt.git
    GIT_TAG        v2.9.1
    GIT_SHALLOW    TRUE
)
set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE)
set(NLOPT_OCTAVE OFF CACHE BOOL "" FORCE)
set(NLOPT_MATLAB OFF CACHE BOOL "" FORCE)
set(NLOPT_GUILE OFF CACHE BOOL "" FORCE)
set(NLOPT_SWIG OFF CACHE BOOL "" FORCE)
set(NLOPT_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(nlopt)

# ---------------------------------------------------------------------------
# 7. Catch2 — Testing framework
#    https://github.com/catchorg/Catch2
# ---------------------------------------------------------------------------
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.7.1
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(Catch2)

# ---------------------------------------------------------------------------
# 8. Lean-VTK — Simple VTK file writer (header + source)
#    https://github.com/mmorse1217/lean-vtk
#    Note: lean-vtk's CMakeLists.txt uses CMAKE_SOURCE_DIR which breaks
#    under FetchContent, so we fetch it and build a small target manually.
# ---------------------------------------------------------------------------
FetchContent_Declare(
    lean_vtk
    GIT_REPOSITORY https://github.com/mmorse1217/lean-vtk.git
    GIT_TAG        master
)
FetchContent_GetProperties(lean_vtk)
if(NOT lean_vtk_POPULATED)
    FetchContent_Populate(lean_vtk)
    # Build a minimal static library from the source
    add_library(leanvtk STATIC
        ${lean_vtk_SOURCE_DIR}/src/leanvtk.cpp
    )
    target_include_directories(leanvtk PUBLIC
        ${lean_vtk_SOURCE_DIR}/include
    )
endif()

# ===========================================================================
# Targets
# ===========================================================================

# --- Main example executable ---
add_executable(scibase_demo src/main.cpp)
target_link_libraries(scibase_demo PRIVATE
    mdspan::mdspan
    Eigen3::Eigen
    fmt::fmt
    pocketfft
    sundials_cvode
    nlopt
)
target_compile_options(scibase_demo PRIVATE -Wall -Wextra -O2)

# --- Tests ---
add_executable(scibase_tests tests/test_main.cpp)
target_link_libraries(scibase_tests PRIVATE
    mdspan::mdspan
    Eigen3::Eigen
    fmt::fmt
    pocketfft
    sundials_cvode
    nlopt
    Catch2::Catch2WithMain
)
target_compile_options(scibase_tests PRIVATE -Wall -Wextra -O2)

include(CTest)
include(Catch)
catch_discover_tests(scibase_tests)
