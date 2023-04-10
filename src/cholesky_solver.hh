#ifndef CHOLESKY_SOLVER_HH
#define CHOLESKY_SOLVER_HH CHOLESKY_SOLVER_HH
#include <memory>
#include <Eigen/SparseCholesky>
#include "linear_operator.hh"
#include "linear_solver.hh"

/** @file cholesky_solver.hh
 *
 * @brief linear solver using (sparse) Cholesky factorisation
 */

/** @class CholeskySolver
 *
 * @brief Solver based on the Eigen sparse Cholesky factorisation
 */
class CholeskySolver : public LinearSolver
{
public:
    /** @brief Create a new instance
     *
     * @param[in]  operator_ underlying linear operator
     */
    CholeskySolver(std::shared_ptr<LinearOperator> linear_operator_);

    /** @brief Solve the linear system Ax = b
     *
     * @param[in] b right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x);

protected:
    /** @brief Expose of sparse matrix to be used */
    typedef LinearOperator::SparseMatrixType SparseMatrixType;
    /** @brief Underlying Eigen solver */
    Eigen::SimplicialLLT<SparseMatrixType> solver;
};

#endif // CHOLESKY_SOLVER_HH