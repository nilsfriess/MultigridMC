#include "diffusion_operator_3d.hh"

/** @file diffusion_operator_3d.cc
 *
 * @brief Implementation of diffusion_operator_3d.hh
 */

/** @brief Create a new instance */
DiffusionOperator3d::DiffusionOperator3d(const std::shared_ptr<Lattice3d> lattice_,
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
    unsigned int nz = lattice_->nz;
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

/** @brief Create a new instance */
MeasuredDiffusionOperator3d::MeasuredDiffusionOperator3d(const std::shared_ptr<Lattice3d> lattice_,
                                                         const std::vector<Eigen::Vector3d> measurement_locations_,
                                                         const Eigen::MatrixXd Sigma_,
                                                         const bool ignore_measurement_cross_correlations_,
                                                         const bool measure_global_,
                                                         const double sigma_average_,
                                                         const double alpha_K_,
                                                         const double beta_K_,
                                                         const double alpha_b_,
                                                         const double beta_b_) : LinearOperator(lattice_,
                                                                                                measurement_locations_.size() + measure_global_)
{
    DiffusionOperator3d diffusion_operator(lattice_,
                                           alpha_K_,
                                           beta_K_,
                                           alpha_b_,
                                           beta_b_);
    A_sparse = diffusion_operator.get_sparse();
    unsigned int nx = lattice_->nx;
    unsigned int ny = lattice_->ny;
    unsigned int nz = lattice_->nz;
    unsigned int nrow = lattice->M;
    unsigned int n_measurements = measurement_locations_.size();
    Sigma_inv = Eigen::MatrixXd(n_measurements + measure_global_, n_measurements + measure_global_);
    Sigma_inv.setZero();
    DenseMatrixType Sigma_local;
    if (ignore_measurement_cross_correlations_)
    {
        Sigma_local = Sigma_.diagonal().asDiagonal();
    }
    else
    {
        Sigma_local = Sigma_;
    }
    Sigma_inv(Eigen::seqN(0, n_measurements), Eigen::seqN(0, n_measurements)) = Sigma_local.inverse();
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplet_list;
    for (int nu = 0; nu < n_measurements; ++nu)
    {
        Eigen::Vector3d x_loc = measurement_locations_[nu];
        int i = int(round(x_loc[0] * nx));
        int j = int(round(x_loc[1] * ny));
        int k = int(round(x_loc[2] * nz));
        unsigned int ell = nx * ny * k + nx * j + i;
        triplet_list.push_back(T(nu, ell, 1.0));
    }
    if (measure_global_)
    {
        for (int j = 0; j < nrow; ++j)
        {
            triplet_list.push_back(T(j, n_measurements, 1. / nrow));
        }
        Sigma_inv(n_measurements, n_measurements) = 1. / sigma_average_;
    }
    B.setFromTriplets(triplet_list.begin(), triplet_list.end());
    Sigma_inv_BT = Sigma_inv.sparseView() * B.transpose();
}

/* Compute posterior mean */
Eigen::VectorXd MeasuredDiffusionOperator3d::posterior_mean(const Eigen::VectorXd &xbar,
                                                            const Eigen::VectorXd &y)
{
    Eigen::SimplicialLLT<SparseMatrixType> solver;
    solver.compute(A_sparse);
    // Compute Bbar = Q^{-1} B
    Eigen::MatrixXd Bbar = solver.solve(B);
    Eigen::VectorXd x_post = xbar + Bbar * (Sigma_inv.inverse() + B.transpose() * Bbar).inverse() * (y - B.transpose() * xbar);
    return x_post;
}
