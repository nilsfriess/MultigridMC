#ifndef SSOR_SAMPLER_HH
#define SSOR_SAMPLER_HH SSOR_SAMPLER_HH
#include <random>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"
#include "smoother/smoother.hh"
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
     */
    SSORSampler(const std::shared_ptr<LinearOperator> linear_operator_,
                std::mt19937_64 &rng_,
                const double omega_) : Base(linear_operator_, rng_),
                                       sor_forward(linear_operator_, rng_, omega_, forward),
                                       sor_backward(linear_operator_, rng_, omega_, backward){};

    /** @brief Carry out a single SOR-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

protected:
    /** @brief Forward smoother */
    const SORSampler sor_forward;
    /** @brief Backward smoother */
    const SORSampler sor_backward;
};

#endif // SSOR_SAMPLER_HH
