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
     * @param[in] rtol_ relative tolerance in solve
     * @param[in] atol_ absolute tolerance in solve
     * @param[in] maxiter_ maximal number of iterations
     * @param[in] verbose_ verbosity level
     */
    LinearSolver(std::shared_ptr<LinearOperator> linear_operator_,
                 const double rtol_ = 1.E-12,
                 const double atol_ = 1.0,
                 const unsigned int maxiter_ = 100,
                 const int verbose_ = 2) : linear_operator(linear_operator_),
                                           rtol(rtol_),
                                           atol(atol_),
                                           maxiter(maxiter_),
                                           verbose(verbose_)
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
    /** @brief relative tolerance for solving */
    const double rtol;
    /** @brief absolute tolerance for solving */
    const double atol;
    /** @brief maximum number of iterations */
    const unsigned int maxiter;
    /** @brief verbosity level */
    const int verbose;
};

#endif // SOLVER_HH