# CMakeLists.txt — VELUGA Post-Processor (VELUA_PP)
#
# Written by Jinsu Rhee (jinsu.rhee@gmail.com)
#
# Structure:
#   VELUA_PP/
#     ├─ CMakeLists.txt
#     └─ src/
#         ├─ main.cxx
#         ├─ global/
#         │   ├─ allvar.h
#         │   └─ config.cxx
#         ├─ pipeline/
#         │   ├─ pipeline.h
#         │   └─ pipeline.cxx
#         └─ utilities/          (future)

cmake_minimum_required(VERSION 3.20)
project(veluga_pp LANGUAGES C CXX)

set(CMAKE_C_STANDARD   11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#-----
# Executable
#-----
add_executable(veluga_pp
    src/main.cxx
    src/getpt.c
    src/global/config.cxx
    src/pipeline/pipeline.cxx
    src/pipeline/rawdata.cxx
    src/pipeline/kdtree.cxx
    src/pipeline/cosmo.cxx
    src/pipeline/miles.cxx
    src/pipeline/bprop.cxx
    src/pipeline/gasprop.cxx
    src/pipeline/save.cxx
)

#-----
# Include path
#-----
target_include_directories(veluga_pp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

#-----
# MPI (required)
#-----
find_package(MPI REQUIRED)
target_compile_definitions(veluga_pp PRIVATE VPP_USE_MPI)
target_link_libraries(veluga_pp PRIVATE MPI::MPI_CXX)

#-----
# OpenMP (required)
#-----
find_package(OpenMP REQUIRED)
target_compile_definitions(veluga_pp PRIVATE VPP_USE_OMP)
target_link_libraries(veluga_pp PRIVATE OpenMP::OpenMP_CXX OpenMP::OpenMP_C)

#-----
# HDF5 (optional — enable when reading RAMSES HDF5 outputs)
#-----
set(HDF5_PREFER_PARALLEL TRUE)
find_package(HDF5 CONFIG QUIET COMPONENTS C CXX)
if (TARGET HDF5::HDF5)
    message(STATUS "HDF5 found (imported target)")
    target_compile_definitions(veluga_pp PRIVATE VPP_USE_HDF5)
    target_link_libraries(veluga_pp PRIVATE HDF5::HDF5)
    target_include_directories(veluga_pp PRIVATE ${HDF5_INCLUDE_DIRS})
else()
    find_package(HDF5 COMPONENTS C CXX QUIET)
    if (HDF5_FOUND)
        message(STATUS "HDF5 found (variables)")
        target_compile_definitions(veluga_pp PRIVATE VPP_USE_HDF5)
        target_link_libraries(veluga_pp PRIVATE ${HDF5_CXX_LIBRARIES} ${HDF5_LIBRARIES})
        target_include_directories(veluga_pp PRIVATE ${HDF5_INCLUDE_DIRS})
    else()
        message(STATUS "HDF5 not found — building without HDF5 support")
    endif()
endif()

#-----
# GCC >= 9 required for std::filesystem
#-----
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)
        message(FATAL_ERROR "GCC >= 9 required for std::filesystem. Found ${CMAKE_CXX_COMPILER_VERSION}.")
    endif()
endif()

#-----
# Git commit hash
#-----
find_package(Git QUIET)
if (GIT_FOUND)
    execute_process(
        COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        OUTPUT_VARIABLE GIT_COMMIT_HASH
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
endif()
if (NOT GIT_COMMIT_HASH)
    set(GIT_COMMIT_HASH "unknown")
endif()
target_compile_definitions(veluga_pp PRIVATE GIT_COMMIT_HASH="${GIT_COMMIT_HASH}")
target_compile_definitions(veluga_pp PRIVATE VPP_TABLE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/table")

#-----
# Optimisation & warnings
#-----
if (MSVC)
    target_compile_options(veluga_pp PRIVATE /O2 /W4)
else()
    target_compile_options(veluga_pp PRIVATE -O2 -Wall -Wextra -Wpedantic)
endif()
