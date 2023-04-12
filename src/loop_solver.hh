#ifndef LOOP_SOLVER_HH
#define LOOP_SOLVER_HH LOOP_SOLVER_HH

#include "linear_operator.hh"
#include "linear_solver.hh"
#include "preconditioner.hh"

/** @file loop_solver.hh
 *
 * @brief A simple preconditioned loop solver.
 *
 */

/** @class LoopSolver
 *
 * @brief Applies the preconditioned iteration
 *
 *   x -> x + P (b-A x)
 *
 * for a given linear operator A and preconditioner P
 */
class LoopSolver : public LinearSolver
{
public:
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] preconditioner_ preconditioner
     * @param[in] rtol_ relative tolerance in solve
     * @param[in] atol_ absolute tolerance in solve
     * @param[in] maxiter_ maximal number of iterations
     * @param[in] verbose_ verbosity level

     */
    LoopSolver(const std::shared_ptr<LinearOperator> linear_operator_,
               std::shared_ptr<Preconditioner> preconditioner_,
               const double rtol_ = 1.E-12,
               const double atol_ = 1.0,
               const unsigned int maxiter_ = 100,
               const int verbose_ = 2) : LinearSolver(linear_operator_,
                                                      rtol_,
                                                      atol_,
                                                      maxiter_,
                                                      verbose_),
                                         preconditioner(preconditioner_)
    {
        r = std::make_shared<SampleState>(linear_operator_->get_lattice()->M);
        Pr = std::make_shared<SampleState>(linear_operator_->get_lattice()->M);
    }

    /** @brief Solve the linear system Ax = b
     *
     * @param[in] b right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x);

protected:
    /** @brief preconditioner */
    std::shared_ptr<Preconditioner> preconditioner;
    /** @brief residual */
    std::shared_ptr<SampleState> r;
    /** @brief P applied to residual */
    std::shared_ptr<SampleState> Pr;
};

#endif // LOOP_SOLVER_HH