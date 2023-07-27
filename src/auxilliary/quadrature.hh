#ifndef QUADRATURE_HH
#define QUADRATURE_HH QUADRATURE_HH

#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include "auxilliary/common.hh"

/** @file quadrature.hh
 *
 * @brief implementation of multi-dimensional Gauss-Legendre quadrature rules
 */

/** @class GaussLegendreQuadrature
 *
 * @brief multi-dimensional Gauss-Legendre quadrature on reference element [0,1]^d
 *
 * Note that for a rule of order n will be able to integrate polynomials
 * of degree 2n+1 exactly.
 */

class GaussLegendreQuadrature
{
public:
    /** @brief Construct a new instance
     *
     * @param dim_ Dimension
     * @param order_ Order of quadrature, currently orders 0,1 and 2 are supported
     */
    GaussLegendreQuadrature(const int dim_, const int order_);

    /** @brief compute quadrature weights */
    std::vector<double> get_weights() const { return weights; }

    /** @brief compute quadrature points */
    std::vector<Eigen::VectorXd> get_points() const { return points; }

protected:
    /** @brief dimension of quadrature rule */
    const int dim;
    /** @brief order of quadrature rule */
    const int order;
    /** @brief vector with quadrature weights */
    std::vector<double> weights;
    /** @brief vector with quadrature points */
    std::vector<Eigen::VectorXd> points;
};

#endif // QUADRATURE_HH