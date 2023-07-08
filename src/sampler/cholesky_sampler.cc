/** @file cholesky_sampler.cc
 *
 * @brief Implementation of cholesky_sampler.hh
 */

#include "cholesky_sampler.hh"

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
        const LinearOperator::SparseMatrixType &B_sparse = linear_operator->get_B();
        const LinearOperator::DenseMatrixType &Sigma_inv = linear_operator->get_Sigma_inv();
        const LinearOperator::SparseMatrixType B_tilde = Sigma_inv.sparseView() * B_sparse.transpose();
        A_sparse += B_sparse * B_tilde;
    }
    LLT_of_A = std::make_shared<LLTType>(A_sparse);
}

/* apply Sampler */
void CholeskySampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const
{
    /* step 1: draw sample xi from normal distribution with zero mean and unit covariance*/
    for (unsigned int ell = 0; ell < xi.size(); ++ell)
    {
        xi[ell] = normal_dist(rng);
    }
    /* step 2: solve U^T g = f */
    Eigen::VectorXd g(xi.size());
    LLT_of_A->solveL(f, g);
    /* step 3: solve U x = xi + g for x */
    LLT_of_A->solveLT(xi + g, x);
}
