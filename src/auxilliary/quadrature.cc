#include "quadrature.hh"

#include <cassert>

/** @file quadrature.cc
 *
 * @brief Implementation of quadrature.hh
 */

/* Construct new instance */
GaussLegendreQuadrature::GaussLegendreQuadrature(const int dim_,
                                                 const int order_) : dim(dim_), order(order_)
{
    assert(dim > 0);
    assert(order >= 0);
    assert(order < 3);
    std::vector<double> weights1d; // Weights for interval [-1,+1]
    std::vector<double> points1d;  // Points on interval [-1,+1]
    switch (order)
    {
    case 0:
        weights1d = {2.0};
        points1d = {0.0};
        break;
    case 1:
        weights1d = {1.0, 1.0};
        points1d = {-1.0 / sqrt(3.0), +1.0 / sqrt(3.0)};
        break;
    case 2:
        weights1d = {5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0};
        points1d = {-sqrt(3.0 / 5.0), 0.0, +sqrt(3.0 / 5.0)};
        break;
    }
    // Construct d-dimensional quadrature weights
    std::vector<std::vector<double>> weights_product = cartesian_product(weights1d, dim);
    for (auto it = weights_product.begin(); it != weights_product.end(); ++it)
    {
        double w = 1.0;
        for (auto sit = it->begin(); sit != it->end(); ++sit)
        {
            w *= 0.5 * (*sit); // scaling factor 1/2 accounts for transform [-1,+1] -> [0,1]
        }
        weights.push_back(w);
    }
    // Construct d-dimensional quadrature points
    std::vector<std::vector<double>> points_product = cartesian_product(points1d, dim);
    for (auto it = points_product.begin(); it != points_product.end(); ++it)
    {
        Eigen::VectorXd p(dim);
        for (int j = 0; j < dim; ++j)
        {
            p[j] = 0.5 * ((*it)[j] + 1.0);
        }
        points.push_back(p);
    }
}
