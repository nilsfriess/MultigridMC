#ifndef SMOOTHER_HH
#define SMOOTHER_HH SMOOTHER_HH
#include <random>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"

/** @file Smoother.hh
 *
 * @brief Base class for smoothers which can be used in multigrid algorithms
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
    virtual void apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const = 0;

protected:
    /** @brief Underlying Linear operator */
    const std::shared_ptr<LinearOperator> linear_operator;
};

/* ******************** factory classes ****************************** */

/** @brief Smoother factory base class */
class SmootherFactory
{
public:
    /** @brief extract a smoother for a given action */
    virtual std::shared_ptr<Smoother> get(std::shared_ptr<LinearOperator> linear_operator) = 0;
};

#endif // SMOOTHER_HH
