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
    for (unsigned int j = 0; j < ny; ++j)
    {
        for (unsigned int i = 0; i < nx; ++i)
        {
            unsigned int ell;
            ell = j * nx + ((i + 0) % nx);
            colidx[ell * stencil_size + 0] = ell;
            ell = ((j + 1 + ny) % ny) * nx + ((i + 0) % nx);
            colidx[ell * stencil_size + 1] = ell;
            ell = j * nx + ((i + 1 + nx) % nx);
            colidx[ell * stencil_size + 2] = ell;
            ell = ((j + 1 + ny) % ny) * nx + ((i + 1 + nx) % nx);
            colidx[ell * stencil_size + 3] = ell;
        }
    }
};