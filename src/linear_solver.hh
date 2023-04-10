#ifndef SOLVER_HH
#define SOLVER_HH SOLVER_HH
#include <memory>
#include "linear_operator.hh"

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
     */
    LinearSolver(std::shared_ptr<LinearOperator> linear_operator_) : linear_operator(linear_operator_) {}

    /** @brief Solve the linear system Ax = b
     *
     * @param[in] b right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x) = 0;

protected:
    /** @brief Underlying linear operator */
    std::shared_ptr<LinearOperator> linear_operator;
};

#endif // SOLVER_HH