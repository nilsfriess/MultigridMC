#include "diffusion_operator_2d.hh"

/** @file diffusion_operator_2d.cc
 *
 * @brief Implementation of diffusion_operator.hh
 */

/** @brief Create a new instance */
DiffusionOperator2d::DiffusionOperator2d(const std::shared_ptr<Lattice2d> lattice_,
                                         const double alpha_K_,
                                         const double beta_K_,
                                         const double alpha_b_,
                                         const double beta_b_) : LinearOperator(lattice_),
                                                                 alpha_K(alpha_K_),
                                                                 beta_K(beta_K_),
                                                                 alpha_b(alpha_b_),
                                                                 beta_b(beta_b_)
{
    unsigned int nx = lattice_->nx;
    unsigned int ny = lattice_->ny;
    double hx = 1. / nx;
    double hy = 1. / ny;
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplet_list;
    unsigned int nrow = lattice->M;
    triplet_list.reserve(5 * nrow);
    for (unsigned int j = 0; j < ny; ++j)
    {
        for (unsigned int i = 0; i < nx; ++i)
        {
            unsigned int ell = nx * j + i;
            unsigned int ell_prime;
            double K_north = K_diff(i * hx, (j + 0.5) * hy);
            double K_south = K_diff(i * hx, (j - 0.5) * hy);
            double K_east = K_diff((i + 0.5) * hx, j * hy);
            double K_west = K_diff((i - 0.5) * hx, j * hy);
            double b_centre = b_zero(i * hx, j * hy);
            // centre
            triplet_list.push_back(T(ell, ell, b_centre + (K_east + K_west) / (hx * hx) + (K_north + K_south) / (hy * hy)));
            // south
            ell_prime = nx * ((j - 1 + ny) % ny) + i;
            triplet_list.push_back(T(ell, ell_prime, -K_south / (hy * hy)));
            // north
            ell_prime = nx * ((j + 1) % ny) + i;
            triplet_list.push_back(T(ell, ell_prime, -K_north / (hy * hy)));
            // west
            ell_prime = nx * j + ((i - 1) % nx);
            triplet_list.push_back(T(ell, ell_prime, -K_west / (hy * hy)));
            // east
            ell_prime = nx * j + ((i + 1 + nx) % nx);
            triplet_list.push_back(T(ell, ell_prime, -K_east / (hy * hy)));
        }
    }
    A_sparse.setFromTriplets(triplet_list.begin(), triplet_list.end());
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

/** @brief Create a new instance */
MeasuredDiffusionOperator2d::MeasuredDiffusionOperator2d(const std::shared_ptr<Lattice2d> lattice_,
                                                         const std::vector<Eigen::Vector2d> measurement_locations_,
                                                         const Eigen::MatrixXd Sigma_,
                                                         const double alpha_K_,
                                                         const double beta_K_,
                                                         const double alpha_b_,
                                                         const double beta_b_) : DiffusionOperator2d(lattice_,
                                                                                                     alpha_K_,
                                                                                                     beta_K_,
                                                                                                     alpha_b_,
                                                                                                     beta_b_),
                                                                                 measurement_locations(measurement_locations_),
                                                                                 Sigma(Sigma_)
{
    unsigned int nx = lattice_->nx;
    unsigned int ny = lattice_->ny;
    Eigen::MatrixXd Sigma_inv = Sigma.inverse();
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplet_list;
    unsigned int nrow = lattice->M;
    unsigned int n_measurements = measurement_locations.size();
    triplet_list.reserve(n_measurements * n_measurements);
    for (int k = 0; k < n_measurements; ++k)
    {
        for (int m = 0; m < n_measurements; ++m)
        {
            Eigen::Vector2d x_loc = measurement_locations[k];
            Eigen::Vector2d y_loc = measurement_locations[m];
            int i_k = int(round(x_loc[0] * nx));
            int j_k = int(round(x_loc[1] * ny));
            int i_m = int(round(y_loc[0] * nx));
            int j_m = int(round(y_loc[1] * ny));
            unsigned int ell_k = nx * j_k + i_k;
            unsigned int ell_m = nx * j_m + i_m;
            triplet_list.push_back(T(ell_k, ell_m, Sigma_inv(k, m)));
        }
    }
    SparseMatrixType A_low_rank(nrow, nrow);
    A_low_rank.setFromTriplets(triplet_list.begin(), triplet_list.end());
    A_sparse += A_low_rank;
}