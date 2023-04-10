#ifndef SAMPLER_HH
#define SAMPLER_HH SAMPLER_HH
#include <random>
#include "linear_operator.hh"
#include "samplestate.hh"

/** @file sampler.hh
 *
 * @brief Monte Carlo samplers
 */

/** @class Sampler
 *
 * @brief Sampler base class */
class Sampler
{
public:
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     */
    Sampler(const LinearOperator &linear_operator_,
            std::mt19937_64 &rng_);

    /** @brief destroy instance */
    ~Sampler()
    {
        delete[] sqrt_inv_diag;
    }

protected:
    /** @brief square root of inverse diagonal matrix entries (required in Gibbs sweep) */
    double *sqrt_inv_diag;

protected:
    /** @brief Underlying Linear operator */
    const LinearOperator &linear_operator;
    /** @brief random number generator */
    std::mt19937_64 &rng;
    /** @brief normal distribution for Gibbs-sweep */
    std::normal_distribution<double> normal_dist;
};

/** @class GibbsSampler
 *
 * @brief Gibbs Sampler
 */
class GibbsSampler : public Sampler
{
public:
    /** @brief Base type*/
    typedef Sampler Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     */
    GibbsSampler(const LinearOperator &linear_operator_,
                 std::mt19937_64 &rng_) : Base(linear_operator_, rng_){};

    /** @brief Carry out a single Gibbs-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    void apply(const std::shared_ptr<SampleState> b,
               std::shared_ptr<SampleState> x);
};

#endif // SAMPLER_HH
