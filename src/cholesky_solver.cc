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
    solver.compute(A);
    B = linear_operator->get_B();
    const DenseMatrixType Sigma_inv = linear_operator->get_Sigma_inv();
    DenseMatrixType Ainv_B(B.rows(), B.cols());
    for (int j = 0; j < B.cols(); ++j)
    {
        Ainv_B(Eigen::seq(0, B.rows() - 1), j) = solver.solve(B(Eigen::seq(0, B.rows() - 1), j));
    }
    B_bar = Ainv_B * (Sigma_inv.inverse() + B.transpose() * Ainv_B).inverse();
}

/** Solve the linear system Ax = b */
void CholeskySolver::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    // Solve the factorised system

    Eigen::VectorXd y = solver.solve(b);
    x = y - B_bar * B.transpose() * y;
}