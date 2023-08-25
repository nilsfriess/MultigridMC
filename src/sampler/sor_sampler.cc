/** @file sor_sampler.cc
 *
 * @brief Implementation of sor_sampler.hh
 */

#include "sor_sampler.hh"

/* Create a new instance */
SORSampler::SORSampler(const std::shared_ptr<LinearOperator> linear_operator_,
                       std::mt19937_64 &rng_,
                       const double omega_,
                       const unsigned int nsmooth_,
                       const Direction direction_) : Base(linear_operator_, rng_),
                                                     omega(omega_),
                                                     direction(direction_),
                                                     nsmooth(nsmooth_),
                                                     c_rhs(linear_operator_->get_ndof()),
                                                     xi(linear_operator_->get_m_lowrank())
{
    const LinearOperator::SparseMatrixType &A_sparse = linear_operator->get_sparse();
    unsigned int nrow = A_sparse.rows();
    sqrt_precision_diag = new double[nrow];
    auto diag = A_sparse.diagonal();
    for (unsigned ell = 0; ell < nrow; ++ell)
    {
        sqrt_precision_diag[ell] = sqrt(diag[ell] * (2. - omega) / omega);
    }
    smoother = std::make_shared<SORSmoother>(linear_operator, omega, 1, direction);
    if (linear_operator->get_m_lowrank())
    {
        Eigen::VectorXd Sigma_inv_diag = linear_operator->get_Sigma_inv().diagonal();
        Sigma_lowrank_inv_sqrt = std::make_shared<Eigen::DiagonalMatrix<double, Eigen::Dynamic>>(Sigma_inv_diag.cwiseSqrt());
    }
}

/* apply Sampler */
void SORSampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const
{
    for (unsigned int k = 0; k < nsmooth; ++k)
    {
        // Diagonal part
        for (unsigned int ell = 0; ell < c_rhs.size(); ++ell)
        {
            double tmp = sqrt_precision_diag[ell];
            c_rhs[ell] = tmp * normal_dist(rng) + f[ell];
        }
        // low-rank correction to covariance matrix
        if (linear_operator->get_m_lowrank() > 0)
        {
            const LinearOperator::SparseMatrixType B = linear_operator->get_B();
            for (unsigned int ell = 0; ell < xi.size(); ++ell)
            {
                xi[ell] = normal_dist(rng);
            }
            c_rhs += B * (*Sigma_lowrank_inv_sqrt) * xi;
        }
        smoother->apply(c_rhs, x);
    }
}
