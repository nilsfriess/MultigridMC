#ifndef SOR_SAMPLER_HH
#define SOR_SAMPLER_HH SOR_SAMPLER_HH
#include <random>
#include <Eigen/Dense>
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
     */
    SORSampler(const std::shared_ptr<LinearOperator> linear_operator_,
               std::mt19937_64 &rng_,
               const double omega_,
               const Direction direction_);

    /** @brief destroy instance */
    ~SORSampler()
    {
        delete[] sqrt_precision_diag;
    }

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
    /** @brief RHS sample */
    mutable Eigen::VectorXd c_rhs;
    /** @brief Low rank correction */
    mutable Eigen::VectorXd xi;
    /** @brief square root of diagonal matrix entries divided by omega */
    double *sqrt_precision_diag;
    /** @brief Underlying smoother */
    std::shared_ptr<SORSmoother> smoother;
    /** @brief Cholesky factorisation U^T U = Sigma^{-1} of low rank covariance matrix */
    std::shared_ptr<LinearOperator::DenseMatrixType> U_lowrank;
};

#endif // SOR_SAMPLER_HH
