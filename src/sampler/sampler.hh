#ifndef SAMPLER_HH
#define SAMPLER_HH SAMPLER_HH
#include <Eigen/Dense>
#include "auxilliary/parallel_random.hh"
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
            std::shared_ptr<RandomGenerator> rng_) : linear_operator(linear_operator_),
                                                     rng(rng_) {}

    /** @brief deep copy
     *
     * Create a deep copy of object, while using a specified random number generator
     *
     * @param[in] random number generator to use
     */
    virtual std::shared_ptr<Sampler> deep_copy(std::shared_ptr<RandomGenerator> rng) = 0;

    /** @brief Draw a new sample x
     *
     * @param[in] f right hand side
     * @param[inout] x new sample
     */
    virtual void
    apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const = 0;

    /** @brief return pointer to underlying linear operator */
    std::shared_ptr<LinearOperator> get_linear_operator() const
    {
        return linear_operator;
    };

    /** @brief fix the right hand side vector f
     *
     * For some samplers this might improve performance.
     *
     * @param[in] f right hand side f that appears in the exponent of the
     *            probability density.
     */
    virtual void fix_rhs(const Eigen::VectorXd &f) {}

    /** @brief unset the right hand side vector g
     *
     * Set the pointer to zero, which will force the solve for g in every
     * call to the apply() method.
     */
    virtual void unfix_rhs() {}

protected:
    /** @brief Underlying Linear operator */
    const std::shared_ptr<LinearOperator> linear_operator;
    /** @brief random number generator */
    std::shared_ptr<RandomGenerator> rng;
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
