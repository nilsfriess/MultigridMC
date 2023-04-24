/** @file sor_sampler.cc
 *
 * @brief Implementation of sor_sampler.hh
 */

#include "sor_sampler.hh"

/* Create a new instance */
SORSampler::SORSampler(const std::shared_ptr<LinearOperator> linear_operator_,
                       std::mt19937_64 &rng_,
                       const double omega_,
                       const Direction direction_) : Base(linear_operator_, rng_),
                                                     omega(omega_),
                                                     direction(direction_),
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
    smoother = std::make_shared<SORSmoother>(linear_operator, omega, direction);
    if (linear_operator->get_m_lowrank())
    {
        Eigen::LLT<LinearOperator::DenseMatrixType,
                   Eigen::Upper>
            LLT_of_Sigma_inv(linear_operator->get_Sigma_inv());
        U_lowrank = std::make_shared<LinearOperator::DenseMatrixType>(LLT_of_Sigma_inv.matrixU());
    }
}