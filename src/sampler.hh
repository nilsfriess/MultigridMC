#ifndef SAMPLER_HH
#define SAMPLER_HH SAMPLER_HH
#include <random>
#include <Eigen/Dense>
#include "linear_operator.hh"

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

    /** @brief Draw a new sample x
     *
     * @param[in] b right hand side
     * @param[inout] x new sample
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) = 0;

protected:
    /** @brief square root of inverse diagonal matrix entries (required in Gibbs sweep) */
    double *sqrt_inv_diag;
    /** @brief Underlying Linear operator */
    const LinearOperator &linear_operator;
    /** @brief random number generator */
    std::mt19937_64 &rng;
    /** @brief normal distribution for Gibbs-sweep */
    std::normal_distribution<double> normal_dist;
};

/** @class Cholesky Sampler
 *
 * @brief Sampler based on (sparse) Cholesky factorisation
 *
 * Given a precision matrix Q, compute the Cholesky factorisation
 * Q = U^T U. Draw a sample xi ~ N(0,I) from a multivariate
 * normal distribution and solve U x = xi. The sample x will then
 * be distributed according to ~exp(-1/2 x^T Q x).
 */

class CholeskySampler : public Sampler
{
public:
    /** @brief Base type*/
    typedef Sampler Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     */
    CholeskySampler(const LinearOperator &linear_operator_,
                    std::mt19937_64 &rng_);

    /** @brief Draw a new sample x
     *
     * @param[in] b right hand side
     * @param[inout] x new sample
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x);

protected:
    typedef Eigen::SimplicialLLT<LinearOperator::SparseMatrixType,
                                 Eigen::Upper,
                                 Eigen::NaturalOrdering<int>>
        LLTType;
    /** @brief Cholesky factorisation */
    std::shared_ptr<LLTType> LLT_of_A;
    /** @brief vector with normal random variables */
    Eigen::VectorXd xi;
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
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x);
};

#endif // SAMPLER_HH
