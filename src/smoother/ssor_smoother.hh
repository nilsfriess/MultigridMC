#ifndef SSOR_SMOOTHER_HH
#define SSOR_SMOOTHER_HH SSOR_SMOOTHER_HH
#include <random>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"
#include "smoother.hh"
#include "sor_smoother.hh"

/** @file ssor_smoother.hh
 *
 * @brief Symmetric successive overrelaxation smoother
 */

/** @class SSORSmoother
 *
 * @brief Symmetric successive overrelaxation smoother with low rank updates
 *
 * This implements the following iteration
 *
 *   x^{k+1/4} = (L +   1/omega * D)^{-1} (b + (L^T - (1-omega)/omega * D) x^k)
 *   y^{k+1/2} = B^T x^{k+1/4}
 *   x^{k+1/2} = x^{k+1/4} - bar(B)_{FW} y^{k+1/2}
 *   x^{k+3/4} = (L^T + 1/omega * D)^{-1} (b + (L - (1-omega)/omega * D) x^{k+1/2})
 *   y^{k+1}   = B^T x^{k+3/4}
 *   x^{k+1}   = x^{k+3/4} - bar(B)_{BW} y^{k+1}
 *
 * Here we have that
 *
 *   bar(B)_{FW} = (L   + 1/omega * D)^{-1} B ( Sigma + B^T (L   + 1/omega * D)^{-1} B )^{-1}
 *   bar(B)_{BW} = (L^T + 1/omega * D)^{-1} B ( Sigma + B^T (L^T + 1/omega * D)^{-1} B )^{-1}
 */
class SSORSmoother : public Smoother
{
public:
    /** @brief Base type*/
    typedef Smoother Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] omega_ overrelaxation factor
     * @param[in] nsmooth_ number of smoothing steps
     *
     */
    SSORSmoother(const std::shared_ptr<LinearOperator> linear_operator_,
                 const double omega_,
                 const int nsmooth_) : Base(linear_operator_),
                                       nsmooth(nsmooth_),
                                       sor_forward(linear_operator_, omega_, 1, forward),
                                       sor_backward(linear_operator_, omega_, 1, backward){};

    /** @brief Carry out a single SOR-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

protected:
    /** @brief number of sweeps */
    const int nsmooth;
    /** @brief Forward smoother */
    const SORSmoother sor_forward;
    /** @brief Backward smoother */
    const SORSmoother sor_backward;
};

/* ******************** factory classes ****************************** */

/** @brief SSOR smoother factor class */
class SSORSmootherFactory : public SmootherFactory
{
public:
    /** @brief Create new instance
     *
     * @param[in] omega_ overrelaxation parameter
     * @param[in] nsmooth_ number of sweeps
     */
    SSORSmootherFactory(const double omega_,
                        const int nsmooth_) : omega(omega_),
                                              nsmooth(nsmooth_) {}

    /** @brief Destructor */
    virtual ~SSORSmootherFactory() {}

    /** @brief Return sampler for a specific  linear operator
     *
     * @param[in] linear_operator_ Underlying linear operator
     */
    virtual std::shared_ptr<Smoother> get(std::shared_ptr<LinearOperator> linear_operator)
    {
        return std::make_shared<SSORSmoother>(linear_operator, omega, nsmooth);
    }

private:
    /** @brief overrelaxation parameter */
    const double omega;
    /** @brief number of sweeps*/
    const int nsmooth;
};

#endif // SSOR_SMOOTHER_HH
