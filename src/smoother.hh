#ifndef SMOOTHER_HH
#define SMOOTHER_HH SMOOTHER_HH
#include <random>
#include <Eigen/Dense>
#include "linear_operator.hh"

/** @file Smoother.hh
 *
 * @brief Smoothers which can be used in multigrid algorithms
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
     */
    Smoother(const std::shared_ptr<LinearOperator> linear_operator_) : linear_operator(linear_operator_){};

    /** @brief Apply smoother once
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) = 0;

protected:
    /** @brief Underlying Linear operator */
    const std::shared_ptr<LinearOperator> linear_operator;
};

/** @class SORSmoother
 *
 * @brief Successive overrelaxation smoother
 */
class SORSmoother : public Smoother
{
public:
    /** @brief Base type*/
    typedef Smoother Base;
    /** @brief Create a new instance
     *
     * @param[in] linear_operator_ underlying linear operator
     * @param[in] omega_ overrelaxation factor
     */
    SORSmoother(const std::shared_ptr<LinearOperator> linear_operator_,
                const double omega_) : Base(linear_operator_), omega(omega_){};

    /** @brief Carry out a single SOR-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x);

protected:
    /** @brief Overrelaxation factor */
    const double omega;
};

/** @class SSORSmoother
 *
 * @brief Symmetric successive overrelaxation smoother
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
     */
    SSORSmoother(const std::shared_ptr<LinearOperator> linear_operator_,
                 const double omega_) : Base(linear_operator_), omega(omega_){};

    /** @brief Carry out a single SOR-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x);

protected:
    /** @brief Overrelaxation factor */
    const double omega;
};

/* ******************** factory classes ****************************** */

/** @brief Smoother factory base class */
class SmootherFactory
{
public:
    /** @brief extract a smoother for a given action */
    virtual std::shared_ptr<Smoother> get(std::shared_ptr<LinearOperator> linear_operator) = 0;
};

/** @brief SOR smoother factor class */
class SORSmootherFactory : public SmootherFactory
{
public:
    /** @brief Create new instance
     *
     * @param[in] omega_ overrelaxation parameter
     */
    SORSmootherFactory(const double omega_) : omega(omega_) {}

    /** @brief Destructor */
    virtual ~SORSmootherFactory() {}

    /** @brief Return sampler for a specific  linear operator
     *
     * @param[in] linear_operator_ Underlying linear operator
     */
    virtual std::shared_ptr<Smoother> get(std::shared_ptr<LinearOperator> linear_operator)
    {
        return std::make_shared<SORSmoother>(linear_operator, omega);
    }

private:
    /** @brief overrelaxation parameter */
    const double omega;
};

/** @brief SSOR smoother factor class */
class SSORSmootherFactory : public SmootherFactory
{
public:
    /** @brief Create new instance
     *
     * @param[in] omega_ overrelaxation parameter
     */
    SSORSmootherFactory(const double omega_) : omega(omega_) {}

    /** @brief Destructor */
    virtual ~SSORSmootherFactory() {}

    /** @brief Return sampler for a specific  linear operator
     *
     * @param[in] linear_operator_ Underlying linear operator
     */
    virtual std::shared_ptr<Smoother> get(std::shared_ptr<LinearOperator> linear_operator)
    {
        return std::make_shared<SSORSmoother>(linear_operator, omega);
    }

private:
    /** @brief overrelaxation parameter */
    const double omega;
};

#endif // SMOOTHER_HH
