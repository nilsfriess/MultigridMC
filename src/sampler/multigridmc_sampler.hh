#ifndef MULTIGRIDMC_SAMPLER_HH
#define MULTIGRIDMC_SAMPLER_HH MULTIGRIDMC_SAMPLER_HH
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"
#include "intergrid/intergrid_operator.hh"
#include "sampler.hh"

/** @file multigridmc_sampler.hh
 *
 * @brief Multigrid Monte Carlo sampler
 */

/** @struct Multigrid Monte Carlo parameters */
struct MultigridMCParameters
{
    /** @brief Number of levels */
    unsigned int nlevel;
    /** @brief Number of presmoothing steps */
    unsigned int npresample;
    /** @brief number of postsmoothing steps */
    unsigned int npostsample;
};

/** @class MultigridMCSampler
 *
 * @brief Sampler based on Multigrid Monte Carlo
 */
class MultigridMCSampler : public Sampler
{
public:
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     * @param[in] params_ multigrid Monte Carlo parameters
     * @param[in] presampler_factory_ factory for presampler on each level
     * @param[in] postsampler_factory_ factory for postsampler on each level
     * @param[in] intergrid_operator_factory_ factory for intergrid operators on each level
     * @param[in] coarse_sampler_factory_ factory for coarse solver
     */
    MultigridMCSampler(std::shared_ptr<LinearOperator> linear_operator_,
                       std::mt19937_64 &rng_,
                       const MultigridMCParameters params_,
                       std::shared_ptr<SamplerFactory> presampler_factory_,
                       std::shared_ptr<SamplerFactory> postsampler_factory_,
                       std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory_,
                       std::shared_ptr<SamplerFactory> coarse_sampler_factory_);

    /** @brief Draw a new sample
     *
     * @param[in] f right hand side b
     * @param[out] x solution x
     */
    virtual void apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const;

protected:
    /** @brief Recursive solve on a givel level
     *
     * @param[in] level level on which to solve recursively
     */
    void sample(const unsigned int level) const;

    /** @brief parameters */
    const MultigridMCParameters params;
    /** @brief presampler factory on each level */
    std::shared_ptr<SamplerFactory> presampler_factory;
    /** @brief postsmoother factory on each level */
    std::shared_ptr<SamplerFactory> postsampler_factory;
    /** @brief intergrid operator factory on each level */
    std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory;
    /** @brief factory for coarse sampler */
    std::shared_ptr<SamplerFactory> coarse_sampler_factory;
    /** @brief coarse level solver */
    std::shared_ptr<Sampler> coarse_sampler;
    /** @brief linear operators on all levels */
    std::vector<std::shared_ptr<LinearOperator>> linear_operators;
    /** @brief smoothers on all levels */
    std::vector<std::shared_ptr<Sampler>> presamplers;
    /** @brief smoothers on all levels */
    std::vector<std::shared_ptr<Sampler>> postsamplers;
    /** @brief intergrid operators on all levels (except the coarsest) */
    std::vector<std::shared_ptr<IntergridOperator>> intergrid_operators;
    /** @brief Solution on each level */
    mutable std::vector<Eigen::VectorXd> x_ell;
    /** @brief RHS on each level */
    mutable std::vector<Eigen::VectorXd> f_ell;
    /** @brief Residual on each level */
    mutable std::vector<Eigen::VectorXd> r_ell;
};

#endif // MULTIGRIDMC_SAMPLER_HH