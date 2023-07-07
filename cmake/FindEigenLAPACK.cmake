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

# Try to compile lapacke code
set(CMAKE_REQUIRED_LINK_OPTIONS ${LAPACK_LINKER_FLAGS} ${LAPACK_LIBRARIES})
check_c_source_compiles("
#include <lapacke.h> 
int main(int argc, char **argv) {
  lapack_int m, n, lda, info;
  double *a, *tau;
  info = LAPACKE_dgeqrf( LAPACK_COL_MAJOR, m, n, a, lda, tau );
}
" LAPACKE_COMPILES)

check_c_source_compiles("
#include <mkl_lapacke.h> 
int main(int argc, char **argv) {
  lapack_int m, n, lda, info;
  double *a, *tau;
  info = LAPACKE_dgeqrf( LAPACK_COL_MAJOR, m, n, a, lda, tau );
}
" MKL_LAPACKE_COMPILES)

if(NOT(LAPACKE_COMPILES OR MKL_LAPACKE_COMPILES))
    message(STATUS "Cannot compile LAPACKE code - trying to find LAPACKE library.")
    find_library(LAPACKE_LIBRARY lapacke)

    if(LAPACKE_LIBRARY)
        message(STATUS "Found LAPACKE: " ${LAPACKE_LIBRARY})
        link_libraries(${LAPACKE_LIBRARY})
    else()
        message(ERROR "Could not find LAPACKE library")
        set(LAPACK_FOUND false)
    endif()
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