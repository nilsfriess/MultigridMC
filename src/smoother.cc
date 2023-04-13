/** @file smoother.cc
 *
 * @brief Implementation of smoother.hh
 */

#include "smoother.hh"

/** apply SOR smoother */
void SORSmoother::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const
{
    const LinearOperator::SparseMatrixType &A_sparse = linear_operator->get_sparse();
    const auto row_ptr = A_sparse.outerIndexPtr();
    const auto col_ptr = A_sparse.innerIndexPtr();
    const auto val_ptr = A_sparse.valuePtr();
    const auto diag_ptr = A_sparse.diagonal();
    unsigned int nrow = A_sparse.rows();
    for (unsigned int ell_ = 0; ell_ < nrow; ++ell_)
    {
        unsigned int ell = (direction == forward) ? ell_ : nrow - 1 - ell_;
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
void SSORSmoother::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const
{
    sor_forward.apply(b, x);
    sor_backward.apply(b, x);
}

/* Create a new instance */
SGSLowRankSmoother::SGSLowRankSmoother(const std::shared_ptr<LinearOperator> linear_operator_) : Base(linear_operator_)
{
    // Smoothers
    sor_smoothers.push_back(SORSmoother(linear_operator, 1.0, forward));
    sor_smoothers.push_back(SORSmoother(linear_operator, 1.0, backward));
    B = linear_operator->get_B();
    LinearOperator::SparseMatrixType A_sparse = linear_operator->get_sparse();
    LinearOperator::DenseMatrixType Sigma = linear_operator->get_Sigma_inv().inverse();
    // Compute bar(B)_{FW} = (L   + D)^{-1} B ( Sigma + B^T (L   + D)^{-1} B )^{-1}
    LinearOperator::DenseMatrixType LD_inv_B = A_sparse.triangularView<Eigen::Lower>().solve(B);
    B_bar.push_back(LD_inv_B * (Sigma + B.transpose() * LD_inv_B).inverse());
    // Compute bar(B)_{BW} = (L^T + D)^{-1} B ( Sigma + B^T (L^T + D)^{-1} B )^{-1}
    LinearOperator::DenseMatrixType LTD_inv_B = A_sparse.triangularView<Eigen::Upper>().solve(B);
    B_bar.push_back(LTD_inv_B * (Sigma + B.transpose() * LTD_inv_B).inverse());
}

/* Carry out a single sweep */
void SGSLowRankSmoother::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const
{
    for (int sweep = 0; sweep < 2; ++sweep)
    {
        // SOR sweep
        sor_smoothers[sweep].apply(b, x);
        // Low-rank update
        x -= B_bar[sweep] * B.transpose() * x;
    }
}