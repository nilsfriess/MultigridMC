/** @file sampler.cc
 *
 * @brief Implementation of sampler.hh
 */

#include "sampler.hh"

/* Create a new instance */
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

/* Create a new instance */
CholeskySampler::CholeskySampler(const LinearOperator &linear_operator_,
                                 std::mt19937_64 &rng_) : Base(linear_operator_,
                                                               rng_),
                                                          xi(linear_operator_.get_ndof())
{
    LinearOperator::SparseMatrixType A_sparse = linear_operator.get_sparse();
    if (linear_operator.get_m_lowrank() > 0)
    {
        // Add contribution from low rank correction
        const LinearOperator::DenseMatrixType &B = linear_operator.get_B();
        const LinearOperator::DenseMatrixType &Sigma_inv = linear_operator.get_Sigma_inv();
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

/* apply Sampler */
void GibbsSampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x)
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
        x[ell] = ((f[ell] - residual) * a_sqrt_inv_diag + normal_dist(rng)) * a_sqrt_inv_diag;
    }
}