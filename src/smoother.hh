#ifndef SMOOTHER_HH
#define SMOOTHER_HH SMOOTHER_HH
#include <random>
#include "linear_operator.hh"
#include "samplestate.hh"

/** @file smoother.hh
 *
 * @brief multigrid smoothers
 */

/** @class Smoother
 *
 * @brief Smoother base class */
class Smoother
{
public:
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     */
    Smoother(const LinearOperator &linear_operator_,
             std::mt19937_64 &rng_);

    /** @brief destroy instance */
    ~Smoother()
    {
        delete[] sqrt_inv_diag;
    }

protected:
    /** @brief square root of inverse diagonal matrix entries (required in Gibbs sweep) */
    double *sqrt_inv_diag;

protected:
    /** @brief Underlying Linear operator */
    const LinearOperator &linear_operator;
    /** @brief random number generator */
    std::mt19937_64 &rng;
    /** @brief normal distribution for Gibbs-sweep */
    std::normal_distribution<double> normal_dist;
};

/** @class GaussSeidelSmoother
 *
 * @brief Gauss Seidel smoother
 */
class GaussSeidelSmoother : public Smoother
{
public:
    /** @brief Base type*/
    typedef Smoother Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] rng_ random number generator
     */
    GaussSeidelSmoother(const LinearOperator &linear_operator_,
                        std::mt19937_64 &rng_) : Base(linear_operator_, rng_){};

    /** @brief Carry out a single Gibbs-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    void apply(const std::shared_ptr<SampleState> b,
               std::shared_ptr<SampleState> x);
};

#endif // SMOOTHER_HH
