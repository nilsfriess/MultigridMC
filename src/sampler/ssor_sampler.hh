#ifndef SSOR_SAMPLER_HH
#define SSOR_SAMPLER_HH SSOR_SAMPLER_HH
#include <random>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"
#include "smoother/sor_smoother.hh"
#include "sampler.hh"
#include "sor_sampler.hh"

/** @file ssor_sampler.hh
 *
 * @brief Monte Carlo samplers based on symmetric successive overrelaxation
 */

/** @class SSORSampler
 *
 * @brief Symmetric successive overrelaxation sampler with low rank updates
 */
class SSORSampler : public Sampler
{
public:
    /** @brief Base type*/
    typedef Sampler Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] omega_ overrelaxation factor
     * @param[in] nsmooth_ number of smoothing steps
     */
    SSORSampler(const std::shared_ptr<LinearOperator> linear_operator_,
                std::mt19937_64 &rng_,
                const double omega_,
                const unsigned int nsmooth_) : Base(linear_operator_, rng_),
                                               omega(omega_),
                                               nsmooth(nsmooth_),
                                               sor_forward(linear_operator_, rng_, omega_, 1, forward),
                                               sor_backward(linear_operator_, rng_, omega_, 1, backward){};

    /** @brief deep copy
     *
     * Create a deep copy of object, while using a specified random number generator
     *
     * @param[in] random number generator to use
     */
    virtual std::shared_ptr<Sampler> deep_copy(std::mt19937_64 &rng)
    {
        std::shared_ptr<LinearOperator> linear_operator_ = linear_operator->deep_copy();
        return std::make_shared<SSORSampler>(linear_operator_,
                                             rng,
                                             omega,
                                             nsmooth);
    };

    /** @brief Carry out a single SOR-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

protected:
    /** @brief smoothing parameter */
    const double omega;
    /** @brief number of smoothing steps*/
    const unsigned int nsmooth;
    /** @brief Forward smoother */
    const SORSampler sor_forward;
    /** @brief Backward smoother */
    const SORSampler sor_backward;
};

/* ******************** factory classes ****************************** */

/** @brief SSOR sampler factory */
class SSORSamplerFactory : public SamplerFactory
{
public:
    /** @brief create a new instance
     *
     * @param[in] rng_ random number generator
     * @param[in] omega_ overrelaxation parameter
     * @param[in] nsmooth_ number of sweeps
     */
    SSORSamplerFactory(std::mt19937_64 &rng_,
                       const double omega_,
                       const int nsmooth_) : rng(rng_),
                                             omega(omega_),
                                             nsmooth(nsmooth_) {}

    /** @brief extract a sampler for a given linear operator
     *
     * @param[in] linear_operator Underlying linear operator
     */
    virtual std::shared_ptr<Sampler> get(std::shared_ptr<LinearOperator> linear_operator)
    {
        return std::make_shared<SSORSampler>(linear_operator, rng, omega, nsmooth);
    };

protected:
    /** @brief random number generator */
    std::mt19937_64 &rng;
    /** @brief Overrelaxation factor */
    const double omega;
    /** @brief number of applications */
    const int nsmooth;
};

#endif // SSOR_SAMPLER_HH
