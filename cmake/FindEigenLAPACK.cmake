# Find BLAS/LAPACK for the Eigen library
#

# find BLAS
find_package(BLAS REQUIRED)

if(NOT BLAS_FOUND)
    message(ERROR "Could not find BLAS library!")
else()
endif()

# find LAPACK
find_package(LAPACK REQUIRED)

if(NOT LAPACK_FOUND)
    message(ERROR "Could not find LAPACK library!")
else()
endif()

# on MACOS we need to explicitly find the LAPACKE library
find_library(LAPACKE_LIBRARY lapacke)

if(LAPACKE_LIBRARY)
    message(STATUS "Found LAPACKE: " ${LAPACKE_LIBRARY})
    link_libraries(${LAPACKE_LIBRARY})
else()
    message(ERROR " Cound not find LAPACKE library")
    set(LAPACK_FOUND false)
endif()

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