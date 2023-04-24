#include "intergrid_operator_2dlinear.hh"

/** @file intergrid_operator_2dlinear.cc
 * @brief Implementation of intergrid_operator-2dlinear.hh
 */

/** @brief Create a new instance */
IntergridOperator2dLinear::IntergridOperator2dLinear(const std::shared_ptr<Lattice2d> lattice_) : Base(lattice_, 9)
{
    // 1d stencil and shift vector
    const double stencil1d[3] = {0.25, 0.5, 0.25};
    const int shift1d[3] = {-1, 0, +1};
    // matrix entries
    int k = 0;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            matrix[k] = stencil1d[i] * stencil1d[j];
            ++k;
        }
    }
    // column indices
    unsigned int nx = lattice_->nx;
    unsigned int ny = lattice_->ny;
    for (unsigned int j = 0; j < ny / 2; ++j)
    {
        for (unsigned int i = 0; i < nx / 2; ++i)
        {
            unsigned int ell_coarse = j * nx / 2 + i;
            unsigned int ell = 2 * j * nx + 2 * i;
            int k = 0;
            for (int i = 0; i < 3; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    Eigen::Vector2i shift = {shift1d[i], shift1d[j]};
                    colidx[ell_coarse * stencil_size + k] = lattice->shift_index(ell, shift);
                    ++k;
                }
            }
        }
    }
};