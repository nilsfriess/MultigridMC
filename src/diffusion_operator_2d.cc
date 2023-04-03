#include "diffusion_operator_2d.hh"

/** @file diffusion_operator_2d.cc
 *
 * @brief Implementation of diffusion_operator.hh
 */

/** @brief Create a new instance */
DiffusionOperator2d::DiffusionOperator2d(const std::shared_ptr<Lattice2d> lattice_,
                                         std::mt19937_64 &rng_,
                                         const double alpha_K_,
                                         const double beta_K_,
                                         const double alpha_b_,
                                         const double beta_b_) : LinearOperator2d5pt(lattice_, rng_),
                                                                 alpha_K(alpha_K_), beta_K(beta_K_), alpha_b(alpha_b_), beta_b(beta_b_)
{
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
            double b_centre = b_zero(i * hx, j * hy);
            matrix[stencil_size * ell + 0] = b_centre + (K_east + K_west) / (hx * hx) + (K_north + K_south) / (hy * hy);
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
    return alpha_K + beta_K * sin(2 * M_PI * x) * sin(2 * M_PI * y);
}

/** @brief Zero order term */
double DiffusionOperator2d::b_zero(const double x, const double y) const
{
    return alpha_b + beta_b * cos(2 * M_PI * x) * cos(2 * M_PI * y);
}