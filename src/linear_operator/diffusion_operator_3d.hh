#ifndef DIFFUSION_OPERATOR_3D_HH
#define DIFFUSION_OPERATOR_3D_HH DIFFUSION_OPERATOR_3D_HH

#include <vector>
#include <Eigen/Dense>
#include "linear_operator.hh"
#include "lattice/lattice3d.hh"

/** @file diffusion_operator_3d.hh
 *
 * @brief Contains class for diffusion operator in three dimensions
 */

/** @class DiffusionOperator3d
 *
 * Two dimensional diffusion operator
 *
 * This is a discretisation of the linear operator defined by
 *
 *   -div( K(x,y,z) grad (u(x,y,z))) + b(x,y,z) u(x,y,z)
 *
 * The diffusion coefficient is assumed to be of the form
 *
 *    K(x,y,z) = alpha_K + beta_K * sin(2 pi x) * sin(2 pi y) * sin(2 pi z)
 *
 * and the zero order term is assumed to be
 *
 *    b(x,y,z) = alpha_b + beta_b * cos(2 pi x) * cos(2 pi y) * cos(2 pi z)
 *
 * for some constants alpha_K, beta_K, alpha_b, beta_b.
 *
 */
class DiffusionOperator3d : public LinearOperator
{
public:
    /** @brief Create a new instance
     *
     * Populates matrix entries across the grid
     *
     * @param[in] lattice_ underlying 3d lattice
     * @param[in] rng_ random number generator
     * @param[in] alpha_K first coefficient in diffusion function
     * @param[in] beta_K second coefficient in diffusion function
     * @param[in] alpha_b first coefficient in zero order term
     * @param[in] beta_b second coefficient in zero order term
     * @param[in] m_lowrank_ the dimension of the low rank correction
     */
    DiffusionOperator3d(const std::shared_ptr<Lattice> lattice_,
                        const double alpha_K_,
                        const double beta_K_,
                        const double alpha_b_,
                        const double beta_b_);

    /** @brief Diffusion coefficient
     *
     * Evaluates the diffusion coefficient at a given position (x,y) in the domain
     *
     * @param[in] x position in x-direction
     * @param[in] y position in y-direction
     * @param[in] z position in z-direction
     */
    double K_diff(const double x, const double y, const double z) const;

    /** @brief Zero order term
     *
     * Evaluates the zero order term at a given position (x,y) in the domain
     *
     * @param[in] x position in x-direction
     * @param[in] y position in y-direction
     * @param[in] z position in z-direction
     */
    double b_zero(const double x, const double y, const double z) const;

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

#endif // DIFFUSION_OPERATOR_3D_HH