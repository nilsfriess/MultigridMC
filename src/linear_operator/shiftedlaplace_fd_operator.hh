#ifndef SHIFTEDLAPLACE_FD_OPERATOR_HH
#define SHIFTEDLAPLACE_FD_OPERATOR_HH SHIFTEDLAPLACE_FD_OPERATOR_HH

#include <vector>
#include <Eigen/Dense>
#include "auxilliary/common.hh"
#include "linear_operator.hh"
#include "lattice/lattice.hh"

/** @file shiftedlaplace_fd_operator.hh
 *
 * @brief Finite difference discretisation of the shifted laplace operator in arbitrary dimensions
 */

/** @class ShiftedLaplaceFDOperator
 *
 * Class for finite difference discretisation of shifted Laplace operator
 *
 *   -alpha_K * div( grad (u)) + alpha_b u
 *
 * with homogeneous Dirichlet boundary conditions
 *
 */
class ShiftedLaplaceFDOperator : public LinearOperator
{
public:
    /** @brief Create a new instance
     *
     * Populates matrix entries across the grid
     *
     * @param[in] lattice_ underlying 2d lattice
     * @param[in] alpha_K coefficient of second order term
     * @param[in] alpha_b coefficient of zero order term
     * @param[in] verbose_ verbosity level
     */
    ShiftedLaplaceFDOperator(const std::shared_ptr<Lattice> lattice_,
                             const double alpha_K_,
                             const double alpha_b_,
                             const int verbose = 0);

protected:
    /** @brief Coefficient of Laplace term */
    const double alpha_K;
    /** @brief Coefficient of zero order term */
    const double alpha_b;
};

#endif // SHIFTEDLAPLACE_FD_OPERATOR_HH