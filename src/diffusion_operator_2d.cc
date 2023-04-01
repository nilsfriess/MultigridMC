#include "diffusion_operator_2d.hh"

/** @file diffusion_operator_2d.cc
 *
 * @brief Implementation of diffusion_operator.hh
 */

/** @brief Create a new instance */
DiffusionOperator2d::DiffusionOperator2d(const Lattice2d &lattice_, std::mt19937_64 &rng_) : LinearOperator2d5pt(lattice_, rng_)
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
            matrix[stencil_size * ell + 0] = 1.0 + (K_east + K_west) / (hx * hx) + (K_north + K_south) / (hy * hy);
            matrix[stencil_size * ell + 1] = -K_south / (hy * hy);
            matrix[stencil_size * ell + 2] = -K_north / (hy * hy);
            matrix[stencil_size * ell + 3] = -K_west / (hy * hy);
            matrix[stencil_size * ell + 4] = -K_east / (hy * hy);
        }
    }
}

/** @brief Diffusion coefficient */
double DiffusionOperator2d::K_diff(const double x, const double y) const
{
    return 0.8 + 0.25 * sin(2 * M_PI * x) * sin(2 * M_PI * y);
}