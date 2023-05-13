# Find Cholmod library
#
# Sets the variable NCHOLDMOD to false, unless Cholmod has been found

find_library(CHOLMOD_LIBRARY cholmod)

if(CHOLMOD_LIBRARY)
    message(STATUS "Found Cholmod library at ${CHOLMOD_LIBRARY}")
    cmake_path(REMOVE_FILENAME CHOLMOD_LIBRARY OUTPUT_VARIABLE CHOLMOD_DIR)
    cmake_path(APPEND CHOLMOD_DIR "..")
    cmake_path(NORMAL_PATH CHOLMOD_DIR)
    find_path(CHOLMOD_INCLUDE_DIR NAMES cholmod.h PATHS ${CHOLMOD_DIR} PATH_SUFFIXES include)
    include_directories(${CHOLMOD_INCLUDE_DIR})
    link_libraries(${CHOLMOD_LIBRARY})
    set(NCHOLMOD "false")
else(CHOLMOD_LIBRARY)
    message(ERROR " Could not find CholMod library! Falling back to Eigen::Simplicial for Cholesky factorisation")
endif(CHOLMOD_LIBRARY)
