#include "intergrid_operator.hh"

/** @file intergrid_operator.cc
 * @brief Implementation of intergrid_operator.hh
 */

/** @brief Create a new instance */
IntergridOperator2dAvg::IntergridOperator2dAvg(const std::shared_ptr<Lattice2d> lattice_) : Base(lattice_, 4)
{
    // matrix entries
    matrix[0] = 1.0;
    matrix[1] = 1.0;
    matrix[2] = 1.0;
    matrix[3] = 1.0;
    // column indices
    unsigned int nx = lattice_->nx;
    unsigned int ny = lattice_->ny;
    for (unsigned int j = 0; j < ny / 2; ++j)
    {
        for (unsigned int i = 0; i < nx / 2; ++i)
        {
            unsigned int ell = j * nx / 2 + i;

            colidx[ell * stencil_size + 0] = 2 * j * nx + 2 * i;
            colidx[ell * stencil_size + 1] = 2 * j * nx + ((2 * i + 1 + nx) % nx);
            colidx[ell * stencil_size + 2] = ((2 * j + 1 + ny) % ny) * nx + 2 * i;
            colidx[ell * stencil_size + 3] = ((2 * j + 1 + ny) % ny) * nx + ((2 * i + 1 + nx) % nx);
        }
    }
};