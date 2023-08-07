#ifndef SHIFTEDLAPLACE_FEM_OPERATOR_HH
#define SHIFTEDLAPLACE_FEM_OPERATOR_HH SHIFTEDLAPLACE_FEM_OPERATOR_HH

#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include "auxilliary/common.hh"
#include "auxilliary/quadrature.hh"
#include "lattice/lattice.hh"
#include "linear_operator.hh"
#include "correlationlength_model.hh"

/** @file shiftedlaplace_fem_operator.hh
 *
 * @brief Finite element discretisation of the shifted laplace operator in arbitrary dimensions
 */

/** @class ShiftedLaplaceFEMOperator
 *
 * Class for shifted Laplace operator in lowest order (multilinear)
 * finite element discretisation. In the continuum the linear operator given by
 *
 *   -( grad (u)) + kappa^{-2} u
 *
 * with homogeneous Dirichlet boundary conditions.
 *
 */
class ShiftedLaplaceFEMOperator : public LinearOperator
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
    ShiftedLaplaceFEMOperator(const std::shared_ptr<Lattice> lattice_,
                              const std::shared_ptr<CorrelationLengthModel> correlationlength_model_,
                              const int verbose = 0);

protected:
    /** @brief evaluate multi-linear basis function in reference cell
     *
     * @param[in] alpha d-dimensional multiindex of basis function, the entries should be 0 or 1
     * @param[in] xhat d-dimensional point in unit reference cell [0,1]^d
     */
    double phi(Eigen::VectorXi alpha, Eigen::VectorXd xhat) const;

    /** @brief evaluate gradients of basis function in reference cell
     *
     * @param[in] alpha d-dimensional multiindex of basis function, the entries should be 0 or 1
     * @param[in] xhat d-dimensional point in unit reference cell [0,1]^d
     */
    Eigen::VectorXd grad_phi(Eigen::VectorXi alpha, Eigen::VectorXd xhat) const;

    /** @brief Model for correlation length in the domain */
    const std::shared_ptr<CorrelationLengthModel> correlationlength_model;
};

#endif // SHIFTEDLAPLACE_FEM_OPERATOR_HH