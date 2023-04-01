#include "linear_operator.hh"
/** @file linear_operator.cc
 *
 * @brief Implementation of linear_operator.hh
 *
 * Defines the specific offsets for operators
 */

/** @brief Offsets in x-direction for 5point operator in 2d */
template <>
const int BaseLinearOperator2d<5, LinearOperator2d5pt>::offset_x[5] = {0, 0, 0, -1, +1};
/** @brief Offsets in y-direction for 5point operator in 2d */
template <>
const int BaseLinearOperator2d<5, LinearOperator2d5pt>::offset_y[5] = {0, -1, +1, 0, 0};

/** @brief Create a new instance */
LinearOperator2d5pt::LinearOperator2d5pt(const std::shared_ptr<Lattice2d> lattice_, std::mt19937_64 &rng_) : Base(lattice_, rng_)
{
    for (unsigned int j = 0; j < ny; ++j)
    {
        for (unsigned int i = 0; i < nx; ++i)
        {
            for (int k = 1; k < stencil_size; ++k)
            {
                unsigned int ell = ((j + offset_y[k] + ny) % ny) * nx + ((i + offset_x[k] + nx) % nx);
                colidx[ell * stencil_size + k] = ell;
            }
        }
    }
}