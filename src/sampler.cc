/** @file sampler.cc
 *
 * @brief Implementation of sampler.hh
 */

#include "sampler.hh"

/** @brief Create a new instance */
Sampler::Sampler(const LinearOperator &linear_operator_,
                 std::mt19937_64 &rng_) : linear_operator(linear_operator_),
                                          rng(rng_),
                                          normal_dist(0.0, 1.0)
{
    const LinearOperator::SparseMatrixType &A_sparse = linear_operator.get_sparse();
    unsigned int nrow = A_sparse.rows();
    sqrt_inv_diag = new double[nrow];
    auto diag = A_sparse.diagonal();
    for (unsigned ell = 0; ell < nrow; ++ell)
    {
        sqrt_inv_diag[ell] = 1 / sqrt(diag[ell]);
    }
}

/** @brief apply Sampler */
void GibbsSampler::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    const LinearOperator::SparseMatrixType &A_sparse = linear_operator.get_sparse();
    const auto row_ptr = A_sparse.outerIndexPtr();
    const auto col_ptr = A_sparse.innerIndexPtr();
    const auto val_ptr = A_sparse.valuePtr();
    unsigned int nrow = A_sparse.rows();
    for (int ell = 0; ell < nrow; ++ell)
    {
        double residual = 0.0;
        for (int k = row_ptr[ell]; k < row_ptr[ell + 1]; ++k)
        {
            unsigned int ell_prime = col_ptr[k];
            residual += val_ptr[k] * x[ell_prime];
        }
        const double a_sqrt_inv_diag = sqrt_inv_diag[ell];
        // subtract diagonal contribution
        residual -= x[ell] / (a_sqrt_inv_diag * a_sqrt_inv_diag);
        x[ell] = ((b[ell] - residual) * a_sqrt_inv_diag + normal_dist(rng)) * a_sqrt_inv_diag;
    }
}