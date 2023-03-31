#ifndef DIFFUSION_OPERATOR_2D_HH
#define DIFFUSION_OPERATOR_2D_HH DIFFUSION_OPERATOR_2D_HH
#include "operator.hh"
#include "lattice2d.hh"

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
    DiffusionOperator2d(const Lattice2d &lattice_, std::mt19937_64 &rng_) : LinearOperator2d5pt(lattice_, rng_)
    {
        unsigned int nx = lattice.nx;
        unsigned int ny = lattice.ny;
        double hx = 1. / nx;
        double hy = 1. / ny;
        for (unsigned int j = 0; j < ny; ++j)
        {
            for (unsigned int i = 0; i < nx; ++i)
            {
                unsigned int ell = nx * j + i;
                double K_north = K_diff(i * hx, (j + 0.5) * hy);
                double K_south = K_diff(i * hx, (j - 0.5) * hy);
                double K_east = K_diff((i + 0.5) * hx, j * hy);
                double K_west = K_diff((i - 0.5) * hx, j * hy);
                data[stencil_size * ell + 0] = 1.0 + (K_east + K_west) / (hx * hx) + (K_north + K_south) / (hy * hy);
                data[stencil_size * ell + 1] = -K_south / (hy * hy);
                data[stencil_size * ell + 2] = -K_north / (hy * hy);
                data[stencil_size * ell + 3] = -K_west / (hy * hy);
                data[stencil_size * ell + 4] = -K_east / (hy * hy);
            }
        }
    }

    /** @brief Diffusion coefficient
     *
     * Evaluates the diffusion coefficient at a given position (x,y) in the domain
     *
     * @param[in] x position in x-direction
     * @param[in] y position in y-direction
     */
    double K_diff(const double x, const double y)
    {
        return 0.8 + 0.25 * sin(2 * M_PI * x) * sin(2 * M_PI * y);
    }

protected:
    using Base::lattice;
};

#endif // DIFFUSION_OPERATOR_2D_HH