#ifndef DIFFUSION_OPERATOR_2D_HH
#define DIFFUSION_OPERATOR_2D_HH DIFFUSION_OPERATOR_2D_HH
#include "linear_operator.hh"
#include "lattice.hh"

/** @file diffusion_operator_2d.hh
 *
 * @brief Contains class for diffusion operator in two dimensions
 */

/** @class DiffusionOperator2d
 *
 * Two dimensional diffusion operator
 */
class DiffusionOperator2d : public LinearOperator2d5pt
{
public:
    /** @brief Create a new instance
     *
     * Populates matrix entries across the grid
     *
     * @param[in] lattice_ underlying 2d lattice
     * @param[in] rng_ random number generator
     */
    DiffusionOperator2d(const std::shared_ptr<Lattice2d> lattice_, std::mt19937_64 &rng_);

    /** @brief Diffusion coefficient
     *
     * Evaluates the diffusion coefficient at a given position (x,y) in the domain
     *
     * @param[in] x position in x-direction
     * @param[in] y position in y-direction
     */
    double K_diff(const double x, const double y) const;

protected:
    using Base::lattice;
};

#endif // DIFFUSION_OPERATOR_2D_HH