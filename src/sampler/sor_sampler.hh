#ifndef SOR_SAMPLER_HH
#define SOR_SAMPLER_HH SOR_SAMPLER_HH
#include <random>
#include <Eigen/Dense>
#include "auxilliary/parallel_random.hh"
#include "linear_operator/linear_operator.hh"
#include "smoother/sor_smoother.hh"
#include "sampler.hh"

/** @file sor_sampler.hh
 *
 * @brief Samplers based on successive overrelaxation
 */

/** @class SORSampler
 *
 * @brief SOR Sampler
 *
 * Sampler based on the matrix splitting M = 1/omega*D+L (forward)
 * or M = 1/omega*D+L^T (backward)
 */
class SORSampler : public Sampler
{
public:
    /** @brief Base type*/
    typedef Sampler Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     * @param[in] omega_ overrelaxation factor
     * @param[in] nsmooth_ number of smoothing steps
     * @param[in] direction_ direction of sampling (forward or backward)
     */
    SORSampler(const std::shared_ptr<LinearOperator> linear_operator_,
               std::shared_ptr<RandomGenerator> rng_,
               const double omega_,
               const unsigned int nsmooth_,
               const Direction direction_);

    /** @brief destroy instance */
    ~SORSampler()
    {
        delete[] sqrt_precision_diag;
    }

    /** @brief deep copy
     *
     * Create a deep copy of object, while using a specified random number generator
     *
     * @param[in] random number generator to use
     */
    virtual std::shared_ptr<Sampler> deep_copy(std::shared_ptr<RandomGenerator> rng)
    {
        std::shared_ptr<LinearOperator> linear_operator_ = linear_operator->deep_copy();
        return std::make_shared<SORSampler>(linear_operator_,
                                            rng,
                                            omega,
                                            nsmooth,
                                            direction);
    };

    /** @brief Carry out a single Gibbs-sweep
     *
     * @param[in] f right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const;

protected:
    /** @brief Overrelaxation factor */
    const double omega;
    /** @brief Sweep direction */
    const Direction direction;
    /** @brief number of smoothing steps */
    const unsigned int nsmooth;
    /** @brief RHS sample */
    mutable Eigen::VectorXd c_rhs;
    /** @brief Low rank correction */
    mutable Eigen::VectorXd xi;
    /** @brief square root of diagonal matrix entries divided by omega */
    double *sqrt_precision_diag;
    /** @brief Underlying smoother */
    std::shared_ptr<SORSmoother> smoother;
    /** @brief Square root = Sigma^{-1/2} of low rank covariance matrix */
    std::shared_ptr<Eigen::DiagonalMatrix<double, Eigen::Dynamic>> Sigma_lowrank_inv_sqrt;
};

/* ******************** factory classes ****************************** */

/** @brief SOR sampler factory */
class SORSamplerFactory : public SamplerFactory
{
public:
    /** @brief create a new instance
     *
     * @param[in] rng_ random number generator
     * @param[in] omega_ overrelaxation parameter
     * @param[in] nsmooth_ number of sweeps
     * @param[in] direction_ sweeping direction (forward or backward)
     */
    SORSamplerFactory(std::shared_ptr<RandomGenerator> rng_,
                      const double omega_,
                      const int nsmooth_,
                      const Direction direction_) : rng(rng_),
                                                    omega(omega_),
                                                    nsmooth(nsmooth_),
                                                    direction(direction_) {}

    /** @brief extract a sampler for a given linear operator
     *
     * @param[in] linear_operator Underlying linear operator
     */
    virtual std::shared_ptr<Sampler> get(std::shared_ptr<LinearOperator> linear_operator)
    {
        return std::make_shared<SORSampler>(linear_operator, rng, omega, nsmooth, direction);
    };

protected:
    /** @brief random number generator */
    std::shared_ptr<RandomGenerator> rng;
    /** @brief Overrelaxation factor */
    const double omega;
    /** @brief number of sweeps */
    const int nsmooth;
    /** @brief Sweep direction */
    const Direction direction;
};

#endif // SOR_SAMPLER_HH
