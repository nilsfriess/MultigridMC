#include "cholesky_solver.hh"
/** @file cholesky_solver.cc
 *
 * @brief Implementation of cholesky_solver.hh
 */

/** Create a new instance */
CholeskySolver::CholeskySolver(std::shared_ptr<LinearOperator> linear_operator_) : LinearSolver(linear_operator_)
{
    // Compute Cholesky factorisation here
    const SparseMatrixType &A = linear_operator->get_sparse();
    solver = std::make_shared<CholmodLLT>(A);
    if (linear_operator->get_m_lowrank() > 0)
    {
        B = linear_operator->get_B();
        const DenseMatrixType Sigma_inv = linear_operator->get_Sigma_inv();
        DenseMatrixType Ainv_B(B.rows(), B.cols());
        Eigen::VectorXd y(B.rows());
        for (int j = 0; j < B.cols(); ++j)
        {
            solver->solve(B(Eigen::seq(0, B.rows() - 1), j), y);
            Ainv_B(Eigen::seq(0, B.rows() - 1), j) = y;
        }
        B_bar = Ainv_B * (Sigma_inv.inverse() + B.transpose() * Ainv_B).inverse();
    }
}

/** Solve the linear system Ax = b */
void CholeskySolver::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    // Solve the factorised system
    if (linear_operator->get_m_lowrank() > 0)
    {
        Eigen::VectorXd y(b.size());
        solver->solve(b, y);
        Eigen::VectorXd BTy = B.transpose() * y;
        x = y - B_bar * BTy;
    }
    else
    {
        solver->solve(b, x);
    }
}