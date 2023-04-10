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
void CholeskySolver::apply(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x)
{
    // Solve the factorised system
    x->data = solver.solve(b->data);
}