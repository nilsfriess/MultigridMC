/** @file cholesky_sampler.cc
 *
 * @brief Implementation of cholesky_sampler.hh
 */

#include "cholesky_sampler.hh"

/* Create a new instance */
SparseCholeskySampler::SparseCholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                                             std::shared_ptr<RandomGenerator> rng_,
                                             const bool verbose_) : Base(linear_operator_, rng_)
{
    LinearOperator::SparseMatrixType A_sparse = linear_operator->get_sparse();
    if (linear_operator->get_m_lowrank() > 0)
    {
        // Add contribution from low rank correction
        const LinearOperator::SparseMatrixType &B_sparse = linear_operator->get_B();
        const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &Sigma_inv = linear_operator->get_Sigma_inv();
        const LinearOperator::SparseMatrixType B_tilde = Sigma_inv * B_sparse.transpose();
        A_sparse += B_sparse * B_tilde;
    }
    LLT_of_A = std::make_shared<SparseLLTType>(A_sparse, verbose_);
}

/* Create a new instance */
DenseCholeskySampler::DenseCholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                                           std::shared_ptr<RandomGenerator> rng_) : Base(linear_operator_, rng_)
{
    LinearOperator::DenseMatrixType A_dense = LinearOperator::DenseMatrixType(linear_operator->get_sparse());
    if (linear_operator->get_m_lowrank() > 0)
    {
        // Add contribution from low rank correction
        const LinearOperator::DenseMatrixType &B = linear_operator->get_B();
        const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &Sigma_inv = linear_operator->get_Sigma_inv();
        A_dense += B * Sigma_inv * B.transpose();
    }
    LLT_of_A = std::make_shared<EigenDenseLLT>(A_dense);
}

/* Create a new instance */
LowRankCholeskySampler::LowRankCholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                                               std::shared_ptr<RandomGenerator> rng_,
                                               const bool verbose_) : Base(linear_operator_, rng_),
                                                                      xi(linear_operator_->get_ndof()),
                                                                      g_rhs(nullptr)
{
    // ==== Step 1 ==== Compute sparse Cholesky factorisation of A
    LinearOperator::SparseMatrixType A_sparse = linear_operator->get_sparse();
    LLT_of_A = std::make_shared<SparseLLTType>(A_sparse, verbose_);
    const unsigned int n = linear_operator->get_ndof();
    const unsigned int m = linear_operator->get_m_lowrank();
    if (m > 0)
    {
        // Include contribution from low rank correction
        const LinearOperator::SparseMatrixType &B = linear_operator->get_B();
        const Eigen::DiagonalMatrix<double, Eigen::Dynamic> &Sigma = linear_operator->get_Sigma();
        // ==== Step 2 ==== Compute V = (v_0,...,v_{m-1}) by solving U^T v_j = b_j
        //         for each column b_j of B = (b_0,...,b_{m-1})
        LinearOperator::DenseMatrixType V(n, m);
        // loop over columns j and solve U^T v_j = b_j
        for (int j = 0; j < m; ++j)
        {
            Eigen::VectorXd v_col(n);
            LLT_of_A->solveL(B.col(j), v_col);
            V.col(j) = v_col;
        }
        // ==== Step 3 ==== Compute the QR decomposition V = QR of V
        Eigen::HouseholderQR<LinearOperator::DenseMatrixType> qr(V);
        Q = std::make_shared<LinearOperator::DenseMatrixType>(qr.householderQ());
        R = std::make_shared<LinearOperator::DenseMatrixType>(Q->transpose() * V);
        // ==== Step 4 ==== Compute the matrix Id - Lambda = Id - R ( Gamma + B A^{-1} B^T)^{-1} R^T
        LinearOperator::DenseMatrixType Id_mxm = LinearOperator::DenseMatrixType::Identity(m, m);
        LinearOperator::DenseMatrixType Id_minus_Lambda = Id_mxm - (*R) * (Sigma.toDenseMatrix() + V.transpose() * V) * R->transpose();
        // ==== Step 5 ==== Compute the (dense) Cholesky factorisation W^T W = Id - Lambda
        Eigen::LLT<LinearOperator::DenseMatrixType, Eigen::Lower> dense_llt(Id_minus_Lambda);

        Q_W_T = std::make_shared<LinearOperator::DenseMatrixType>((*Q) * (dense_llt.matrixL().toDenseMatrix() - Id_mxm));
        Q_W = std::make_shared<LinearOperator::DenseMatrixType>((*Q) * (dense_llt.matrixU().toDenseMatrix() - Id_mxm));
    }
}

/* Draw a new sample */
void LowRankCholeskySampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const
{
    // ==== Step 1 ==== draw sample xi from normal distribution with zero mean and unit covariance
    for (unsigned int ell = 0; ell < xi.size(); ++ell)
    {
        xi[ell] = rng->draw_normal();
    }
    std::shared_ptr<Eigen::VectorXd> g = g_rhs;
    if (g == nullptr)
    {
        // ==== Step 2 ==== solve U^T g = f
        g = std::make_shared<Eigen::VectorXd>(xi.size());
        LLT_of_A->solveL(f, *g);
        // ==== Step 3 ==== Set g -> g + Q (W-Id) (Q^T g)
        if (linear_operator->get_m_lowrank())
        {
            Eigen::VectorXd Q_Tg = Q->transpose() * (*g);
            (*g) += (*Q_W) * Q_Tg;
        }
    }
    // ==== Step 4 ==== Add zeta to xi
    xi += (*g);
    // ==== Step 5 ==== Set xi -> xi + Q (W^T-Id) (Q^T g)
    if (linear_operator->get_m_lowrank())
    {
        Eigen::VectorXd Q_Txi = Q->transpose() * (xi);
        xi += (*Q_W_T) * Q_Txi;
    }
    /* ==== Step 6 ==== solve U x = xi for x */
    LLT_of_A->solveLT(xi, x);
}

/* fix the right hand side vector g from a given f */
void LowRankCholeskySampler::fix_rhs(const Eigen::VectorXd &f)
{

    g_rhs = std::make_shared<Eigen::VectorXd>(f.size());
    // ==== Step 1 ==== solve U^T g = f
    LLT_of_A->solveL(f, *g_rhs);
    // ==== Step 2 ==== Set g -> g + Q (W-Id) (Q^T g)
    if (linear_operator->get_m_lowrank())
    {
        Eigen::VectorXd Q_Tg = Q->transpose() * (*g_rhs);
        (*g_rhs) += (*Q_W) * Q_Tg;
    }
}