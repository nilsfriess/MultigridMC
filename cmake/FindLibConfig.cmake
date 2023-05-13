# Find libconfig
find_library(LIBCONFIG_LIBRARY config++)

if(LIBCONFIG_LIBRARY)
    message(STATUS "Found libconfig library at ${LIBCONFIG_LIBRARY}")
    cmake_path(REMOVE_FILENAME LIBCONFIG_LIBRARY OUTPUT_VARIABLE LIBCONFIG_DIR)
    cmake_path(APPEND LIBCONFIG_DIR "..")
    cmake_path(NORMAL_PATH LIBCONFIG_DIR)
    find_path(LIBCONFIG_INCLUDE_DIR NAMES libconfig.hh PATHS ${LIBCONFIG_DIR})
    include_directories(${LIBCONFIG_INCLUDE_DIR})
    link_libraries(${LIBCONFIG_LIBRARY})
else(LIBCONFIG_LIBRARY)
    message(FATAL_ERROR "Cound not find libconfig library!")
endif(LIBCONFIG_LIBRARY)
