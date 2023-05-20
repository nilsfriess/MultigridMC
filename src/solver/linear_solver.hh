#ifndef SOLVER_HH
#define SOLVER_HH SOLVER_HH
#include <memory>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"

/** @file linear_solver.hh
 *
 * @brief Linear solver base class
 */

/** @class LinearSolver
 *
 * @brief base class for linear solvers
 */
class LinearSolver
{
public:
    /** @brief Create a new instance
     *
     * @param[in] operator_ underlying linear operator
     * @param[in] params_ solver parameters
     */
    LinearSolver(std::shared_ptr<LinearOperator> linear_operator_) : linear_operator(linear_operator_)
    {
    }

    /** @brief Solve the linear system Ax = b
     *
     * @param[in] b right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) = 0;

protected:
    /** @brief Underlying linear operator */
    std::shared_ptr<LinearOperator> linear_operator;
};

/* ******************** factory classes ****************************** */

/** @brief Linear solver factory base class */
class LinearSolverFactory
{
public:
    /** @brief extract a linear solver for a given action */
    virtual std::shared_ptr<LinearSolver> get(std::shared_ptr<LinearOperator> linear_operator) = 0;
};

#endif // SOLVER_HH