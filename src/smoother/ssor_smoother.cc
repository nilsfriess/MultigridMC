/** @file ssor_smoother.cc
 *
 * @brief Implementation of ssor_smoother.hh
 */

#include "ssor_smoother.hh"

/** apply SSOR smoother */
void SSORSmoother::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x) const
{
    for (int k = 0; k < nsmooth; ++k)
    {
        sor_forward.apply(b, x);
        sor_backward.apply(b, x);
    }
}