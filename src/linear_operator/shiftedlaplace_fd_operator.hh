#ifndef SHIFTEDLAPLACE_FD_OPERATOR_HH
#define SHIFTEDLAPLACE_FD_OPERATOR_HH SHIFTEDLAPLACE_FD_OPERATOR_HH

#include <vector>
#include <Eigen/Dense>
#include "auxilliary/common.hh"
#include "linear_operator.hh"
#include "correlationlength_model.hh"
#include "lattice/lattice.hh"

/** @file shiftedlaplace_fd_operator.hh
 *
 * @brief Finite difference discretisation of the shifted laplace operator in arbitrary dimensions
 */

/** @class ShiftedLaplaceFDOperator
 *
 * Class for finite difference discretisation of shifted Laplace operator
 *
 *   -div( grad (u)) + kappa^2 u
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
     * @param[in] correlationlength_model_ model for correlation length
     * @param[in] verbose_ verbosity level
     */
    ShiftedLaplaceFDOperator(const std::shared_ptr<Lattice> lattice_,
                             const std::shared_ptr<CorrelationLengthModel> correlationlength_model_,
                             const int verbose = 0);

protected:
    /** @brief Correlation length model */
    std::shared_ptr<CorrelationLengthModel> correlationlength_model;
};

#endif // SHIFTEDLAPLACE_FD_OPERATOR_HH