#ifndef QUANTITYOFINTEREST_HH
#define QUANTITYOFINTEREST_HH QUANTITYOFINTEREST_HH
#include <memory>
#include <Eigen/Dense>
#include "linear_operator/linear_operator.hh"

/** @file quantityofinterest.hh
 * @brief Header file for quantities of interest
 */

/** @class QoI
 *
 * @brief Abstract base class for state-dependent quantity of interest
 *
 */
class QoI
{
public:
  /** @brief Create new instance
   *
   */
  QoI() {}

  /** @brief Evaluate on a state
   *
   * @param[in] x State on which to evaluate the QoI
   */
  const Eigen::VectorXd virtual evaluate(const Eigen::VectorXd &x) = 0;
};

/** Base class for QoI factory */
class QoIFactory
{
public:
  /** @brief extract a qoi for a given action */
  virtual std::shared_ptr<QoI> get(const std::shared_ptr<LinearOperator> linear_operator) = 0;
};

#endif // QUANTITYOFINTEREST_HH
