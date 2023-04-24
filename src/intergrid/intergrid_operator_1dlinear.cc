#include "intergrid_operator_1dlinear.hh"

/** @file intergrid_operator_1dlinear.cc
 * @brief Implementation of intergrid_operator_1dlinear.hh
 */

/** @brief Create a new instance */
IntergridOperator1dLinear::IntergridOperator1dLinear(const std::shared_ptr<Lattice1d> lattice_) : Base(lattice_, 3)
{
    // 1d stencil and shift vector
    const double stencil1d[3] = {0.25, 0.5, 0.25};
    const int shift1d[3] = {-1, 0, +1};
    // matrix entries
    for (int i = 0; i < 3; ++i)
    {
        matrix[i] = stencil1d[i];
    }
    // column indices
    for (unsigned int i = 0; i < lattice_->n / 2; ++i)
    {
        unsigned int ell_coarse = i;
        unsigned int ell = 2 * i;
        for (int i = 0; i < 3; ++i)
        {
            Eigen::VectorXi shift(1);
            shift[0] = shift1d[i];
            colidx[ell_coarse * stencil_size + i] = lattice->shift_index(ell, shift);
        }
    }
};