#include "cholesky_solver.hh"
/** @file cholesky_solver.cc
 *
 * @brief Implementation of cholesky_solver.hh
 */

/** Create a new instance */
CholeskySolver::CholeskySolver(std::shared_ptr<LinearOperator> linear_operator_) : LinearSolver(linear_operator_)
{
    // Compute Cholesky factorisation here
    const SparseMatrixType &A = linear_operator->as_sparse();
    solver.compute(A);
}

/** Solve the linear system Ax = b */
void CholeskySolver::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    // Solve the factorised system
    x = solver.solve(b);
}