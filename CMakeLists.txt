cmake_minimum_required(VERSION 3.22)

project(MultigridMC)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undefined-var-template")

set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

cmake_policy(SET CMP0135 NEW)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Define user options
option(USE_CHOLMOD "Use Cholmod for Cholesky factorisation?" ON)

# use thorough testing?
option(USE_THOROUGH_TESTS "Use thorough tests?" ON)

# find BLAS
find_package(BLAS REQUIRED)

if(BLAS_FOUND)
  message(STATUS "Found BLAS library")
else(BLAS_FOUND)
  message(ERROR "Could not find BLAS library!")
endif(BLAS_FOUND)

# find LAPACK
find_package(LAPACK REQUIRED)

if(LAPACK_FOUND)
  message(STATUS "Found LAPACK library")
else(LAPACK_FOUND)
  message(ERROR "Could not find LAPACK library!")
endif(LAPACK_FOUND)

# Set up Eigen with BLAS/LAPACK support
if(BLAS_FOUND AND LAPACK_FOUND)
  message(STATUS "Using Eigen with BLAS/LAPACK support.")
  link_libraries(BLAS::BLAS)
  link_libraries(LAPACK::LAPACK)
  set(EIGEN_USE_BLAS "true")
  set(EIGEN_USE_LAPACKE "true")
else()
  message(WARNING "Falling back to Eigen without BLAS/LAPACK support.")
endif()

find_library(LAPACKE_LIBRARY lapacke)

if(LAPACKE_LIBRARY)
  message(STATUS "Found LAPACKE library at " ${LAPACKE_LIBRARY})
  link_libraries(${LAPACKE_LIBRARY})
else()
  message(STATUS "Cound not find lapacke")
endif()

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Find Cholmod
set(NCHOLMOD "true")

if(USE_CHOLMOD)
  include(FindCholMod)
endif(USE_CHOLMOD)

# Find libconfig library
include(FindLibConfig)

# Initialize global list of cc files
set(SOURCES "" CACHE INTERNAL "sources")

include_directories(src)

add_subdirectory(src)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
