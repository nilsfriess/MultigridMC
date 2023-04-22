/** @file sampler.cc
 *
 * @brief Implementation of sampler.hh
 */

#include "sampler.hh"

/* Create a new instance */
CholeskySampler::CholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                                 std::mt19937_64 &rng_) : Base(linear_operator_,
                                                               rng_),
                                                          xi(linear_operator_->get_ndof())
{
    LinearOperator::SparseMatrixType A_sparse = linear_operator->get_sparse();
    if (linear_operator->get_m_lowrank() > 0)
    {
        // Add contribution from low rank correction
        const LinearOperator::DenseMatrixType &B = linear_operator->get_B();
        const LinearOperator::DenseMatrixType &Sigma_inv = linear_operator->get_Sigma_inv();
        const LinearOperator::SparseMatrixType B_sparse = B.sparseView();
        const LinearOperator::SparseMatrixType B_tilde = Sigma_inv.sparseView() * B_sparse.transpose();
        A_sparse += B_sparse * B_tilde;
    }
    LLT_of_A = std::make_shared<LLTType>(A_sparse);
}

/* apply Sampler */
void CholeskySampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x)
{
    /* step 1: draw sample xi from normal distribution with zero mean and unit covariance*/
    for (unsigned int ell = 0; ell < xi.size(); ++ell)
    {
        xi[ell] = normal_dist(rng);
    }
    /* step 2: solve U^T g = f */
    auto L_triangular = LLT_of_A->matrixL();
    Eigen::VectorXd g = L_triangular.solve(f);
    /* step 3: solve U x = xi + g for x */
    auto U_triangular = LLT_of_A->matrixU();
    x = U_triangular.solve(xi + g);
}

/* Create a new instance */
SORSampler::SORSampler(const std::shared_ptr<LinearOperator> linear_operator_,
                       std::mt19937_64 &rng_,
                       const double omega_,
                       const Direction direction_) : Base(linear_operator_, rng_),
                                                     omega(omega_),
                                                     direction(direction_),
                                                     b_rhs(linear_operator_->get_ndof())
{
    const LinearOperator::SparseMatrixType &A_sparse = linear_operator->get_sparse();
    unsigned int nrow = A_sparse.rows();
    sqrt_diag_over_omega = new double[nrow];
    auto diag = A_sparse.diagonal();
    for (unsigned ell = 0; ell < nrow; ++ell)
    {
        sqrt_diag_over_omega[ell] = sqrt(diag[ell] * (2. - omega) / omega);
    }
    smoother = std::make_shared<SORSmoother>(linear_operator, omega, direction);
}

/* apply Sampler */
void SORSampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x)
{
    for (unsigned int ell = 0; ell < b_rhs.size(); ++ell)
    {
        double tmp = sqrt_diag_over_omega[ell];
        b_rhs[ell] = tmp * normal_dist(rng) + f[ell];
    }
    smoother->apply(b_rhs, x);
}
