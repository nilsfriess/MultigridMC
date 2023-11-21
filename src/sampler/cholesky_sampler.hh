#ifndef CHOLESKY_SAMPLER_HH
#define CHOLESKY_SAMPLER_HH CHOLESKY_SAMPLER_HH
#include <random>
#include <Eigen/Dense>
#include "auxilliary/parallel_random.hh"
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
                    std::shared_ptr<RandomGenerator> rng_) : Base(linear_operator_, rng_),
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
            xi[ell] = rng->draw_normal();
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
                          std::shared_ptr<RandomGenerator> rng_,
                          const bool verbose_ = false);

    /** @brief deep copy
     *
     * Create a deep copy of object, while using a specified random number generator
     *
     * @param[in] random number generator to use
     */
    virtual std::shared_ptr<Sampler> deep_copy(std::shared_ptr<RandomGenerator> rng)
    {
        const std::shared_ptr<LinearOperator> linear_operator_ = linear_operator->deep_copy();
        return std::make_shared<SparseCholeskySampler>(linear_operator_, rng);
    };

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
                         std::shared_ptr<RandomGenerator> rng_);

    /** @brief deep copy
     *
     * Create a deep copy of object, while using a specified random number generator
     *
     * @param[in] random number generator to use
     */
    virtual std::shared_ptr<Sampler> deep_copy(std::shared_ptr<RandomGenerator> rng)
    {
        const std::shared_ptr<LinearOperator> linear_operator_ = linear_operator->deep_copy();
        return std::make_shared<DenseCholeskySampler>(linear_operator_, rng);
    };
};

/** @class LowRankCholeskySampler
 *
 * @brief Class for sampler based on Cholesky factorisation with low-rank correction
 *
 *  Setup:
 *      1. Compute the (sparse) Cholesky factorisation U^T U = A of the system matrix A
 *      2. Compute the n x m matrix V by solving U^T V = B
 *      3. Compute the QR decomposition V = QR of V where Q is a n x m matrix with
 *         orthonormal columns and R is a m x m matrix
 *      4. Compute the m x m matrix Lambda = R ( Sigma + V^T V)^{-1} R^T
 *      5. Compute the (dense) Cholesky factorisation W^T W = Id - Lambda of Id - Lambda
 *
 *  Computation of RHS (can be done in setup if f is fixed):
 *      1. Given the RHS f, solve the triangular U^T g = f for g
 *      2. Compute zeta = g + Q_{W} (Q^T g) with Q_{W} := Q (W - Id)
 *
 *  Sampling:
 *      1. Draw multivariate normal xi ~ N(zeta, Id)
 *      2. Set eta = xi + Q_{W^T} (Q^T xi) with Q_{W^T} := Q (W^T - Id)
 *      3. Solve the triangular system U x = eta for x
 */

class LowRankCholeskySampler : public Sampler
{
public:
    /** @brief Base type*/
    typedef Sampler Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     */
    LowRankCholeskySampler(const std::shared_ptr<LinearOperator> linear_operator_,
                           std::shared_ptr<RandomGenerator> rng_,
                           const bool verbose_ = false);

    /** @brief Draw a new sample x
     *
     * @param[in] f right hand side
     * @param[inout] x new sample
     */
    virtual void apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const;

    /** @brief fix the right hand side vector g from a given f
     *
     * Compute g from given RHS  by solving U^T g = f
     * once. This will then avoid the repeated solution of this triangular
     * system whenever apply() is called.
     *
     * @param[in] f right hand side f that appears in the exponent of the
     *            probability density.
     */
    virtual void fix_rhs(const Eigen::VectorXd &f);

    /** @brief unfix the right hand side vector g
     *
     * Set the pointer to zero, which will force the solve for g in every
     * call to the apply() method.
     */
    virtual void unfix_rhs()
    {
        g_rhs = nullptr;
    }

    /** @brief deep copy
     *
     * Create a deep copy of object, while using a specified random number generator
     *
     * @param[in] random number generator to use
     */
    virtual std::shared_ptr<Sampler> deep_copy(std::shared_ptr<RandomGenerator> rng)
    {
        const std::shared_ptr<LinearOperator> linear_operator_ = linear_operator->deep_copy();
        return std::make_shared<LowRankCholeskySampler>(linear_operator_, rng);
    };

protected:
    /** @brief Cholesky factorisation */
    std::shared_ptr<SparseLLTType> LLT_of_A;
    /** @brief vector with normal random variables */
    mutable Eigen::VectorXd xi;
    /** @brief first modified right hand side vector */
    mutable std::shared_ptr<Eigen::VectorXd> g_rhs;
    /** @brief second modified right hand side vector */
    mutable std::shared_ptr<Eigen::VectorXd> zeta;
    /** @brief the small m x m matrix that arises in the QR factorisation of V = U^{-T} B*/
    std::shared_ptr<LinearOperator::DenseMatrixType> R;
    /** @brief the n x m matrix that arises in the QR factorisation of V = U^{-T} B;
     * the columns of Q form an orthonormal system */
    std::shared_ptr<LinearOperator::DenseMatrixType> Q;
    /** @brief the matrix Q (W^T - Id) */
    std::shared_ptr<LinearOperator::DenseMatrixType> Q_W_T;
    /** @brief the matrix Q (W - Id) */
    std::shared_ptr<LinearOperator::DenseMatrixType> Q_W;
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
    SparseCholeskySamplerFactory(std::shared_ptr<RandomGenerator> rng_) : rng(rng_) {}

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
    std::shared_ptr<RandomGenerator> rng;
};

/** @brief Cholesky sampler factory */
class DenseCholeskySamplerFactory : public SamplerFactory
{
public:
    /** @brief create a new instance
     *
     * @param[in] rng_ random number generator
     */
    DenseCholeskySamplerFactory(std::shared_ptr<RandomGenerator> rng_) : rng(rng_) {}

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
    std::shared_ptr<RandomGenerator> rng;
};

#endif // CHOLESKY_SAMPLER_HH
