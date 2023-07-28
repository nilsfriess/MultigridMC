#ifndef DIFFUSION_OPERATOR_HH
#define DIFFUSION_OPERATOR_HH DIFFUSION_OPERATOR_HH

#include <chrono>
#include <vector>
#include <Eigen/Dense>
#include "auxilliary/common.hh"
#include "auxilliary/quadrature.hh"
#include "linear_operator.hh"
#include "lattice/lattice.hh"

/** @file diffusion_operator.hh
 *
 * @brief Contains class for diffusion operator in arbitrary dimensions
 */

/** @class DiffusionOperator
 *
 * Base class for diffusion operator in lowest order (multilinear)
 * finite element discretisation. In the continuum the linear operator given by
 *
 *   -div( K grad (u)) + b u
 *
 */
class DiffusionOperator : public LinearOperator
{
public:
    /** @brief Create a new instance
     *
     * Populates matrix entries across the grid
     *
     * @param[in] lattice_ underlying 2d lattice
     * @param[in] rng_ random number generator
     * @param[in] alpha_K first coefficient in diffusion function
     * @param[in] beta_K second coefficient in diffusion function
     * @param[in] alpha_b first coefficient in zero order term
     * @param[in] beta_b second coefficient in zero order term
     * @param[in] verbose_ verbosity level
     */
    DiffusionOperator(const std::shared_ptr<Lattice> lattice_,
                      const double alpha_K_,
                      const double beta_K_,
                      const double alpha_b_,
                      const double beta_b_,
                      const int verbose = 0);

protected:
    /** @brief Diffusion coefficient
     *
     * Evaluates the diffusion coefficient at a given position (x,y) in the domain
     *
     * @param[in] x position in x-direction
     * @param[in] y position in y-direction
     */
    double K_diff(const Eigen::VectorXd x) const;

    /** @brief Zero order term
     *
     * Evaluates the zero order term at a given position (x,y) in the domain
     *
     * @param[in] x position in x-direction
     * @param[in] y position in y-direction
     */
    double b_zero(const Eigen::VectorXd x) const;

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

    /** @brief First coefficient in diffusion function */
    const double alpha_K;
    /** @brief Second coefficient in diffusion function */
    const double beta_K;
    /** @brief First coefficient in zero order term */
    const double alpha_b;
    /** @brief Second coefficient in zero order term */
    const double beta_b;
};

#endif // DIFFUSION_OPERATOR_HH