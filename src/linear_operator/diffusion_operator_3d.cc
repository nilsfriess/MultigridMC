#include "diffusion_operator_3d.hh"

/** @file diffusion_operator_3d.cc
 *
 * @brief Implementation of diffusion_operator_3d.hh
 */

/** @brief Create a new instance */
DiffusionOperator3d::DiffusionOperator3d(const std::shared_ptr<Lattice> lattice_,
                                         const double alpha_K_,
                                         const double beta_K_,
                                         const double alpha_b_,
                                         const double beta_b_) : LinearOperator(lattice_),
                                                                 alpha_K(alpha_K_),
                                                                 beta_K(beta_K_),
                                                                 alpha_b(alpha_b_),
                                                                 beta_b(beta_b_)
{
    Eigen::VectorXi shape = lattice_->shape();
    unsigned int nx = shape[0];
    unsigned int ny = shape[1];
    unsigned int nz = shape[2];
    double hx = 1. / nx;
    double hy = 1. / ny;
    double hz = 1. / nz;
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplet_list;
    unsigned int nrow = lattice->M;
    triplet_list.reserve(7 * nrow);
    for (unsigned int k = 0; k < nz; ++k)
    {
        for (unsigned int j = 0; j < ny; ++j)
        {
            for (unsigned int i = 0; i < nx; ++i)
            {
                unsigned int ell = nx * ny * k + nx * j + i;
                unsigned int ell_prime;
                double K_north = K_diff(i * hx, (j + 0.5) * hy, k * hz);
                double K_south = K_diff(i * hx, (j - 0.5) * hy, k * hz);
                double K_east = K_diff((i + 0.5) * hx, j * hy, k * hz);
                double K_west = K_diff((i - 0.5) * hx, j * hy, k * hz);
                double K_up = K_diff(i * hx, j * hy, (k + 0.5) * hz);
                double K_down = K_diff(i * hx, j * hy, (k - 0.5) * hz);
                double b_centre = b_zero(i * hx, j * hy, k * hz);
                // centre
                triplet_list.push_back(T(ell, ell, b_centre * hx * hy * hz + (K_east + K_west) * hy * hz / hx + (K_north + K_south) * hx * hz / hy + (K_up + K_down) * hx * hy / hz));
                // south
                ell_prime = nx * ny * k + nx * ((j - 1 + ny) % ny) + i;
                triplet_list.push_back(T(ell, ell_prime, -K_south * hx * hz / hy));
                // north
                ell_prime = nx * ny * k + nx * ((j + 1) % ny) + i;
                triplet_list.push_back(T(ell, ell_prime, -K_north * hx * hz / hy));
                // west
                ell_prime = nx * ny * k + nx * j + ((i - 1) % nx);
                triplet_list.push_back(T(ell, ell_prime, -K_west * hy * hz / hx));
                // east
                ell_prime = nx * ny * k + nx * j + ((i + 1 + nx) % nx);
                triplet_list.push_back(T(ell, ell_prime, -K_east * hy * hz / hx));
                // up
                ell_prime = nx * ny * ((k + 1 + nz) % nz) + nx * j + i;
                triplet_list.push_back(T(ell, ell_prime, -K_up * hx * hy / hz));
                // down
                ell_prime = nx * ny * ((k - 1 + nz) % nz) + nx * j + i;
                triplet_list.push_back(T(ell, ell_prime, -K_down * hx * hy / hz));
            }
        }
    }
    A_sparse.setFromTriplets(triplet_list.begin(), triplet_list.end());
}

/** @brief Diffusion coefficient */
double DiffusionOperator3d::K_diff(const double x, const double y, const double z) const
{
    return alpha_K + beta_K * sin(2 * M_PI * x) * sin(2 * M_PI * y) * sin(2 * M_PI * z);
}

/** @brief Zero order term */
double DiffusionOperator3d::b_zero(const double x, const double y, const double z) const
{
    return alpha_b + beta_b * cos(2 * M_PI * x) * cos(2 * M_PI * y) * cos(2 * M_PI * z);
}
