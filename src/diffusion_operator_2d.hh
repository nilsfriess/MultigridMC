#ifndef DIFFUSION_OPERATOR_2D_HH
#define DIFFUSION_OPERATOR_2D_HH DIFFUSION_OPERATOR_2D_HH

#include <vector>
#include "linear_operator.hh"
#include "lattice.hh"

/** @file diffusion_operator_2d.hh
 *
 * @brief Contains class for diffusion operator in two dimensions
 */

/** @class DiffusionOperator2d
 *
 * Two dimensional diffusion operator
 *
 * This is a discretisation of the linear operator defined by
 *
 *   -div( K(x,y) grad (u(x,y))) + b(x,y) u(x,y)
 *
 * The diffusion coefficient is assumed to be of the form
 *
 *    K(x,y) = alpha_K + beta_K * sin(2 pi x) * sin(2 pi y)
 *
 * and the zero order term is assumed to be
 *
 *    b(x,y) = alpha_b + beta_b * cos(2 pi x) * cos(2 pi y)
 *
 * for some constants alpha_K, beta_K, alpha_b, beta_b.
 *
 */
class DiffusionOperator2d : public LinearOperator
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
     */
    DiffusionOperator2d(const std::shared_ptr<Lattice2d> lattice_,
                        const double alpha_K_ = 0.8,
                        const double beta_K_ = 0.2,
                        const double alpha_b_ = 0.9,
                        const double beta_b_ = 0.1);

    /** @brief Diffusion coefficient
     *
     * Evaluates the diffusion coefficient at a given position (x,y) in the domain
     *
     * @param[in] x position in x-direction
     * @param[in] y position in y-direction
     */
    double K_diff(const double x, const double y) const;

    /** @brief Zero order term
     *
     * Evaluates the zero order term at a given position (x,y) in the domain
     *
     * @param[in] x position in x-direction
     * @param[in] y position in y-direction
     */
    double b_zero(const double x, const double y) const;

protected:
    /** @brief First coefficient in diffusion function */
    const double alpha_K;
    /** @brief Second coefficient in diffusion function */
    const double beta_K;
    /** @brief First coefficient in zero order term */
    const double alpha_b;
    /** @brief Second coefficient in zero order term */
    const double beta_b;
};

/** @brief diffusion operator with measurements */
class MeasuredDiffusionOperator2d : public DiffusionOperator2d
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
     * @param[in] measurement_locations_ coordinates of locations where the field is measured
     * @param[in] Sigma_ covariance matrix of measurements
     */
    MeasuredDiffusionOperator2d(const std::shared_ptr<Lattice2d> lattice_,
                                const std::vector<Eigen::Vector2d> measurement_locations_,
                                const Eigen::MatrixXd Sigma_,
                                const double alpha_K_ = 0.8,
                                const double beta_K_ = 0.2,
                                const double alpha_b_ = 0.9,
                                const double beta_b_ = 0.1);

protected:
    /** @brief measurement locations */
    const std::vector<Eigen::Vector2d> measurement_locations;
    /** @brief variances of measurements */
    const Eigen::MatrixXd Sigma;
};

#endif // DIFFUSION_OPERATOR_2D_HH