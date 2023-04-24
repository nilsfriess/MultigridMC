#ifndef LOOP_SOLVER_HH
#define LOOP_SOLVER_HH LOOP_SOLVER_HH

#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"
#include "linear_solver.hh"
#include "preconditioner/preconditioner.hh"
#include "iterative_solver.hh"

/** @file loop_solver.hh
 *
 * @brief Loop solver (preconditioned Richardson iteration)
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
        r = Eigen::VectorXd(linear_operator_->get_lattice()->M);
        Pr = Eigen::VectorXd(linear_operator_->get_lattice()->M);
    }

    /** @brief Solve the linear system Ax = b
     *
     * @param[in] b right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x);

protected:
    /** @brief preconditioner */
    std::shared_ptr<Preconditioner> preconditioner;
    /** @brief residual */
    Eigen::VectorXd r;
    /** @brief P applied to residual */
    Eigen::VectorXd Pr;
};

#endif // LOOP_SOLVER_HH
