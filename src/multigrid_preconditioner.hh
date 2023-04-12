#ifndef MULTIGRID_PRECONDITIONER_HH
#define MULTIGRID_PRECONDITIONER_HH MULTIGRID_PRECONDITIONER_HH
#include <memory>
#include <vector>
#include <Eigen/SparseCholesky>
#include "linear_operator.hh"
#include "intergrid_operator.hh"
#include "preconditioner.hh"
#include "smoother.hh"

/** @file multigrid_preconditioner.hh
 *
 * @brief multigrid preconditioner
 */

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
     * @param[in] number of levels
     * @param[in] smoother_factory_ factory for smoothers on each level
     * @param[in] intergrid_operator_factory_ factory for intergrid operators on each level
     */
    MultigridPreconditioner(std::shared_ptr<LinearOperator> linear_operator_,
                            const unsigned int nlevel_,
                            std::shared_ptr<SmootherFactory> smoother_factory_,
                            std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory_);

    /** @brief Solve the linear system Ax = b
     *
     * @param[in] b right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x);

protected:
    /** @brief Recursive solve on a givel level
     *
     * @param[in] level level on which to solve recursively
     */
    void solve(const unsigned int level);

    /** @brief number of levels */
    const unsigned int nlevel;
    /** @brief smoother factory on each level */
    std::shared_ptr<SmootherFactory> smoother_factory;
    /** @brief intergrid operator factory on each level */
    std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory;
    /** @brief linear operators on all levels */
    std::vector<std::shared_ptr<LinearOperator>> linear_operators;
    /** @brief smoothers on all levels */
    std::vector<std::shared_ptr<Smoother>> smoothers;
    /** @brief intergrid operators on all levels (except the coarsest) */
    std::vector<std::shared_ptr<IntergridOperator>> intergrid_operators;
    /** @brief Solution on each level */
    std::vector<std::shared_ptr<SampleState>> x_ell;
    /** @brief RHS on each level */
    std::vector<std::shared_ptr<SampleState>> b_ell;
    /** @brief Residual on each level */
    std::vector<std::shared_ptr<SampleState>> r_ell;
};

#endif // MULTIGRID_PRECONDITIONER_HH