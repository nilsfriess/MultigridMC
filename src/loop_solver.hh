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
class LoopSolver : public IterativeSolver
{
public:
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] preconditioner_ preconditioner
     * @param[in] params_ linear solver parameters

     */
    LoopSolver(const std::shared_ptr<LinearOperator> linear_operator_,
               std::shared_ptr<Preconditioner> preconditioner_,
               const IterativeSolverParameters params_) : IterativeSolver(linear_operator_, params_),
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