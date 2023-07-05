#ifndef CHOLESKY_SAMPLER_HH
#define CHOLESKY_SAMPLER_HH CHOLESKY_SAMPLER_HH
#include <random>
#include <Eigen/Dense>
#include "auxilliary/cholesky_wrapper.hh"
#include "linear_operator/linear_operator.hh"
#include "sampler.hh"

/** @file cholesky_sampler.hh
 *
 * @brief Sampler based on sparse Cholesky factorisation
 */

/** @class Cholesky Sampler
 *
 * @brief Sampler based on (sparse) Cholesky factorisation
 *
 * Given a precision matrix Q, compute the Cholesky factorisation
 * Q = U^T U. Then draw an independent sample from the distribution
 * pi(x) = N exp(-1/2 x^T Q x + f^T x) in three steps:
 *
 *   1. Draw a sample xi ~ N(0,I) from a multivariate normal distribution
 *      with mean 0 and variance I
 *   2. solve the triangular system U^T g = f for g
 *   3. solve the triangular U x = xi + g for x.
 *
 */

class SparseCholeskySampler : public Sampler
{
public:
    /** @brief Base type*/
    typedef Sampler Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     */
    SparseCholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                          std::mt19937_64 &rng_);

    /** @brief Draw a new sample x
     *
     * @param[in] f right hand side
     * @param[inout] x new sample
     */
    virtual void apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const;

protected:
#ifndef NCHOLMOD
    /** @brief Use Cholmod's supernodal Cholesky factorisation */
    typedef CholmodLLT LLTType;
#else  // NCHOLMOD
    /** @brief Use Eigen's native Cholesky factorisation */
    typedef EigenSimplicialLLT LLTType;
#endif // NCHOLMOD
    /** @brief Cholesky factorisation */
    std::shared_ptr<LLTType> LLT_of_A;
    /** @brief vector with normal random variables */
    mutable Eigen::VectorXd xi;
};

/* ******************** factory classes ****************************** */

/** @brief Cholesky sampler factory */
class SparseCholeskySamplerFactory : public SamplerFactory
{
public:
    /** @brief create a new instance
     *
     * @param[in] rng_ random number generator
     */
    SparseCholeskySamplerFactory(std::mt19937_64 &rng_) : rng(rng_) {}

    /** @brief extract a sampler for a given linear operator
     *
     * @param[in] linear_operator Underlying linear operator
     */
    virtual std::shared_ptr<Sampler> get(std::shared_ptr<LinearOperator> linear_operator)
    {
        return std::make_shared<SparseCholeskySampler>(linear_operator, rng);
    };

protected:
    /** @brief random number generator */
    std::mt19937_64 &rng;
};

#endif // CHOLESKY_SAMPLER_HH
