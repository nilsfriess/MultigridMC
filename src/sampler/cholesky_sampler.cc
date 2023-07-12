/** @file cholesky_sampler.cc
 *
 * @brief Implementation of cholesky_sampler.hh
 */

#include "cholesky_sampler.hh"

/* Create a new instance */
SparseCholeskySampler::SparseCholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                                             std::mt19937_64 &rng_,
                                             const bool verbose_) : Base(linear_operator_, rng_)
{
    LinearOperator::SparseMatrixType A_sparse = linear_operator->get_sparse();
    if (linear_operator->get_m_lowrank() > 0)
    {
        // Add contribution from low rank correction
        const LinearOperator::SparseMatrixType &B_sparse = linear_operator->get_B();
        const LinearOperator::DenseMatrixType &Sigma_inv = linear_operator->get_Sigma_inv();
        const LinearOperator::SparseMatrixType B_tilde = Sigma_inv.sparseView() * B_sparse.transpose();
        A_sparse += B_sparse * B_tilde;
    }
    LLT_of_A = std::make_shared<SparseLLTType>(A_sparse, verbose_);
}

/* Create a new instance */
DenseCholeskySampler::DenseCholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                                           std::mt19937_64 &rng_) : Base(linear_operator_, rng_)
{
    LinearOperator::DenseMatrixType A_dense = LinearOperator::DenseMatrixType(linear_operator->get_sparse());
    if (linear_operator->get_m_lowrank() > 0)
    {
        // Add contribution from low rank correction
        const LinearOperator::DenseMatrixType &B = linear_operator->get_B();
        const LinearOperator::DenseMatrixType &Sigma_inv = linear_operator->get_Sigma_inv();
        A_dense += B * Sigma_inv * B.transpose();
    }
    LLT_of_A = std::make_shared<EigenDenseLLT>(A_dense);
}
