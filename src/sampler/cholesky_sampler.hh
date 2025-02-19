#ifndef CHOLESKY_SAMPLER_HH
#define CHOLESKY_SAMPLER_HH CHOLESKY_SAMPLER_HH
#include <random>
#include <Eigen/Dense>
#include "auxilliary/cholesky_wrapper.hh"
#include "linear_operator/linear_operator.hh"
#include "sampler.hh"

/** @file cholesky_sampler.hh
 *
 * @brief Samplers based on Cholesky factorisation
 *
 * * Given a precision matrix Q, compute the Cholesky factorisation
 * Q = U^T U. Then draw an independent sample from the distribution
 * pi(x) = N exp(-1/2 x^T Q x + f^T x) in three steps:
 *
 *   1. Draw a sample xi ~ N(0,I) from a multivariate normal distribution
 *      with mean 0 and variance I
 *   2. solve the triangular system U^T g = f for g
 *   3. solve the triangular U x = xi + g for x.
 */

/** @class Sparse Cholesky Sampler
 *
 * @brief Class for sampler based on sparse Cholesky factorisation
 *
 */

template <typename LLTType>
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
    CholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                    std::mt19937_64 &rng_) : Base(linear_operator_, rng_),
                                             xi(linear_operator_->get_ndof()),
                                             g_rhs(nullptr) {}

    /** @brief Draw a new sample x
     *
     * @param[in] f right hand side
     * @param[inout] x new sample
     */
    virtual void apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const
    {
        /* step 1: draw sample xi from normal distribution with zero mean and unit covariance*/
        for (unsigned int ell = 0; ell < xi.size(); ++ell)
        {
            xi[ell] = normal_dist(rng);
        }
        std::shared_ptr<Eigen::VectorXd> g = g_rhs;
        if (g == nullptr)
        {
            /* step 2: solve U^T g = f */
            g = std::make_shared<Eigen::VectorXd>(xi.size());
            LLT_of_A->solveL(f, *g);
        }
        /* step 3: solve U x = xi + g for x */
        LLT_of_A->solveLT(xi + (*g), x);
    }

    /** @brief fix the right hand side vector g from a given f
     *
     * Compute g from given RHS  by solving U^T g = f
     * once. This will then avoid the repeated solution of this triangular
     * system whenever apply() is called.
     *
     * @param[in] f right hand side f that appears in the exponent of the
     *            probability density.
     */
    virtual void fix_rhs(const Eigen::VectorXd &f)
    {
        /* step 2: solve U^T g = f */
        g_rhs = std::make_shared<Eigen::VectorXd>(f.size());
        LLT_of_A->solveL(f, *g_rhs);
    }

    /** @brief unfix the right hand side vector g
     *
     * Set the pointer to zero, which will force the solve for g in every
     * call to the apply() method.
     */
    virtual void unfix_rhs()
    {
        g_rhs = nullptr;
    }

protected:
    /** @brief Cholesky factorisation */
    std::shared_ptr<LLTType> LLT_of_A;
    /** @brief vector with normal random variables */
    mutable Eigen::VectorXd xi;
    /** @brief modified right hand side vector */
    mutable std::shared_ptr<Eigen::VectorXd> g_rhs;
};

#ifndef NCHOLMOD
/** @brief Use Cholmod's supernodal Cholesky factorisation */
typedef CholmodLLT SparseLLTType;
#else  // NCHOLMOD
/** @brief Use Eigen's native Cholesky factorisation */
typedef EigenSimplicialLLT SparseLLTType;
#endif // NCHOLMOD
class SparseCholeskySampler : public CholeskySampler<SparseLLTType>
{
public:
    /** @brief Base type*/
    typedef CholeskySampler<SparseLLTType> Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     * @param[in] verbose_ print out additional information?
     */
    SparseCholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                          std::mt19937_64 &rng_,
                          const bool verbose_ = false);

protected:
    using Base::xi;
};

/** @class Dense Cholesky Sampler
 *
 * @brief Class for sampler based on dense Cholesky factorisation
 *
 */
class DenseCholeskySampler : public CholeskySampler<EigenDenseLLT>
{
public:
    /** @brief Base type*/
    typedef CholeskySampler<EigenDenseLLT> Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     */
    DenseCholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                         std::mt19937_64 &rng_);
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

/** @brief Cholesky sampler factory */
class DenseCholeskySamplerFactory : public SamplerFactory
{
public:
    /** @brief create a new instance
     *
     * @param[in] rng_ random number generator
     */
    DenseCholeskySamplerFactory(std::mt19937_64 &rng_) : rng(rng_) {}

    /** @brief extract a sampler for a given linear operator
     *
     * @param[in] linear_operator Underlying linear operator
     */
    virtual std::shared_ptr<Sampler> get(std::shared_ptr<LinearOperator> linear_operator)
    {
        return std::make_shared<DenseCholeskySampler>(linear_operator, rng);
    };

protected:
    /** @brief random number generator */
    std::mt19937_64 &rng;
};

#endif // CHOLESKY_SAMPLER_HH
