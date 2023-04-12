/** @file smoother.cc
 *
 * @brief Implementation of smoother.hh
 */

#include "smoother.hh"

/** apply SOR smoother */
void SORSmoother::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    const LinearOperator::SparseMatrixType &A_sparse = linear_operator->as_sparse();
    const auto row_ptr = A_sparse.outerIndexPtr();
    const auto col_ptr = A_sparse.innerIndexPtr();
    const auto val_ptr = A_sparse.valuePtr();
    const auto diag_ptr = A_sparse.diagonal();
    unsigned int nrow = A_sparse.rows();
    for (int ell = 0; ell < nrow; ++ell)
    {
        double residual = 0.0;
        for (int k = row_ptr[ell]; k < row_ptr[ell + 1]; ++k)
        {
            unsigned int ell_prime = col_ptr[k];
            residual += val_ptr[k] * x[ell_prime];
        }
        x[ell] += omega * (b[ell] - residual) / diag_ptr[ell];
    }
}

/** apply SSOR smoother */
void SSORSmoother::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    const LinearOperator::SparseMatrixType &A_sparse = linear_operator->as_sparse();
    const auto row_ptr = A_sparse.outerIndexPtr();
    const auto col_ptr = A_sparse.innerIndexPtr();
    const auto val_ptr = A_sparse.valuePtr();
    const auto diag_ptr = A_sparse.diagonal();
    unsigned int nrow = A_sparse.rows();
    for (int sweep = 0; sweep < 2; ++sweep)
    {
        for (int ell_ = 0; ell_ < nrow; ++ell_)
        {
            // Count up in sweep 0 and down in sweep 1
            unsigned int ell = (sweep == 0) ? ell_ : nrow - 1 - ell_;
            double residual = 0.0;
            for (int k = row_ptr[ell]; k < row_ptr[ell + 1]; ++k)
            {
                unsigned int ell_prime = col_ptr[k];
                residual += val_ptr[k] * x[ell_prime];
            }
            x[ell] += omega * (b[ell] - residual) / diag_ptr[ell];
        }
    }
}