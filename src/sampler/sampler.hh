#ifndef SAMPLER_HH
#define SAMPLER_HH SAMPLER_HH
#include <random>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"
#include "smoother/smoother.hh"

/** @file sampler.hh
 *
 * @brief Monte Carlo samplers
 */

/** @class Sampler
 *
 * @brief Sampler base class
 *
 * These samplers are used to sample from Gaussian distributions with a given
 * precision matrix Q and mean Q^{-1} f, for which the probability density is
 *
 *   pi(x) = N exp(-1/2 x^T Q x + f^T x)
 *
 * where N is a normalisation constant.*/
class Sampler
{
public:
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     */
    Sampler(const std::shared_ptr<LinearOperator> linear_operator_,
            std::mt19937_64 &rng_) : linear_operator(linear_operator_),
                                     rng(rng_),
                                     normal_dist(0.0, 1.0) {}

    /** @brief Draw a new sample x
     *
     * @param[in] f right hand side
     * @param[inout] x new sample
     */
    virtual void apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const = 0;

    /** @brief return pointer to underlying linear operator */
    std::shared_ptr<LinearOperator> get_linear_operator() const
    {
        return linear_operator;
    };

protected:
    /** @brief Underlying Linear operator */
    const std::shared_ptr<LinearOperator> linear_operator;
    /** @brief random number generator */
    std::mt19937_64 &rng;
    /** @brief normal distribution for Gibbs-sweep */
    mutable std::normal_distribution<double> normal_dist;
};

/* ******************** factory classes ****************************** */

/** @brief Sampler factory base class */
class SamplerFactory
{
public:
    /** @brief extract a sampler for a given linear operator
     *
     * @param[in] linear_operator Underlying linear operator
     */
    virtual std::shared_ptr<Sampler> get(std::shared_ptr<LinearOperator> linear_operator) = 0;
};

#endif // SAMPLER_HH
