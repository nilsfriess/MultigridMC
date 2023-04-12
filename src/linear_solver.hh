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
    virtual void apply(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x) = 0;

protected:
    /** @brief Underlying linear operator */
    std::shared_ptr<LinearOperator> linear_operator;
};

/** @class IterativeSolverParameters
 *
 * @brief iterative linear solver parameters
 */
struct IterativeSolverParameters
{
    /** @brief relative tolerance for solving */
    double rtol;
    /** @brief absolute tolerance for solving */
    double atol;
    /** @brief maximum number of iterations */
    unsigned int maxiter;
    /** @brief verbosity level */
    int verbose;
};

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
    virtual void apply(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x) = 0;

protected:
    /** @brief Solver Parameters */
    const IterativeSolverParameters params;
};

#endif // SOLVER_HH