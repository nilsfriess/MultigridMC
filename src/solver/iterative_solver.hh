#ifndef ITERATIVE_SOLVER_HH
#define ITERATIVE_SOLVER_HH ITERATIVE_SOLVER_HH

#include <Eigen/Dense>
#include "auxilliary/parameters.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_solver.hh"
#include "preconditioner/preconditioner.hh"
#include "linear_solver.hh"

/** @file iterative_solver.hh
 *
 * @brief Iterative solver base class
 *
 */

/** @class IterativeSolver
 *
 * @brief base class for iterative solvers
 */
class IterativeSolver : public LinearSolver
{
public:
    /** @brief Create a new instance
     *
     * @param[in] operator_ underlying linear operator
     * @param[in] params_ solver parameters
     */
    IterativeSolver(std::shared_ptr<LinearOperator> linear_operator_,
                    const IterativeSolverParameters params_) : LinearSolver(linear_operator_),
                                                               params(params_)
    {
    }

    /** @brief Solve the linear system Ax = b
     *
     * @param[in] b right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) = 0;

protected:
    /** @brief Solver Parameters */
    const IterativeSolverParameters params;
};

#endif // ITERATIVE_SOLVER_HH
