cmake_minimum_required(VERSION 3.22)

project(MultigridMC)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-undefined-var-template")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(libconfig 1.7.3 REQUIRED NO_MODULE)

find_library(CHOLMOD_LIBRARY cholmod)

if(CHOLMOD_LIBRARY)
  message(STATUS "Found Cholmod library at ${CHOLMOD_LIBRARY}")
  cmake_path(REMOVE_FILENAME CHOLMOD_LIBRARY OUTPUT_VARIABLE CHOLMOD_DIR)
  cmake_path(APPEND CHOLMOD_DIR "..")
  cmake_path(NORMAL_PATH CHOLMOD_DIR)
  find_path(CHOLMOD_INCLUDE_DIR NAMES cholmod.h PATHS ${CHOLMOD_DIR})
  include_directories(${CHOLMOD_INCLUDE_DIR})
else(CHOLMOD_LIBRARY)
  message(FATAL_ERROR "Cound not find Cholmod library!")
endif(CHOLMOD_LIBRARY)

find_library(CONFIG++_LIBRARY config++)

if(CONFIG++_LIBRARY)
  message(STATUS "Found libconfig++ library at ${CONFIG++_LIBRARY}")
else(CONFIG++_LIBRARY)
  message(FATAL_ERROR "Cound not find libconfig++ library!")
endif(CONFIG++_LIBRARY)

if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

# Initialize global list of cc files
set(SOURCES "" CACHE INTERNAL "sources")

include_directories(src)

add_subdirectory(src)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
