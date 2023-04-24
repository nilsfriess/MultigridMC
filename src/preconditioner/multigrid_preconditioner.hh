#ifndef MULTIGRID_PRECONDITIONER_HH
#define MULTIGRID_PRECONDITIONER_HH MULTIGRID_PRECONDITIONER_HH
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"
#include "intergrid/intergrid_operator.hh"
#include "solver/linear_solver.hh"
#include "preconditioner.hh"
#include "smoother/smoother.hh"

/** @file multigrid_preconditioner.hh
 *
 * @brief multigrid preconditioner
 */

/** @struct multigrid parameters */
struct MultigridParameters
{
    /** @brief Number of levels */
    unsigned int nlevel;
    /** @brief Number of presmoothing steps */
    unsigned int npresmooth;
    /** @brief number of postsmoothing steps */
    unsigned int npostsmooth;
};

/** @class MultigridPreconditioner
 *
 * @brief Preconditioner based on the multigrid algorithm
 */
class MultigridPreconditioner : public Preconditioner
{
public:
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] params_ multigrid parameters
     * @param[in] smoother_factory_ factory for smoothers on each level
     * @param[in] intergrid_operator_factory_ factory for intergrid operators on each level
     * @param[in] coarse_solver_factory_ factory for coarse solver
     */
    MultigridPreconditioner(std::shared_ptr<LinearOperator> linear_operator_,
                            const MultigridParameters params_,
                            std::shared_ptr<SmootherFactory> smoother_factory_,
                            std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory_,
                            std::shared_ptr<LinearSolverFactory> coarse_solver_factory_);

    /** @brief Solve the linear system Ax = b
     *
     * @param[in] b right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x);

protected:
    /** @brief Recursive solve on a givel level
     *
     * @param[in] level level on which to solve recursively
     */
    void solve(const unsigned int level);

    /** @brief parameters */
    const MultigridParameters params;
    /** @brief smoother factory on each level */
    std::shared_ptr<SmootherFactory> smoother_factory;
    /** @brief intergrid operator factory on each level */
    std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory;
    /** @brief factory for coarse solver */
    std::shared_ptr<LinearSolverFactory> coarse_solver_factory;
    /** @brief coarse level solver */
    std::shared_ptr<LinearSolver> coarse_solver;
    /** @brief linear operators on all levels */
    std::vector<std::shared_ptr<LinearOperator>> linear_operators;
    /** @brief smoothers on all levels */
    std::vector<std::shared_ptr<Smoother>> smoothers;
    /** @brief intergrid operators on all levels (except the coarsest) */
    std::vector<std::shared_ptr<IntergridOperator>> intergrid_operators;
    /** @brief Solution on each level */
    std::vector<Eigen::VectorXd> x_ell;
    /** @brief RHS on each level */
    std::vector<Eigen::VectorXd> b_ell;
    /** @brief Residual on each level */
    std::vector<Eigen::VectorXd> r_ell;
};

#endif // MULTIGRID_PRECONDITIONER_HH