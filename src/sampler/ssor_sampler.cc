/** @file ssor_sampler.cc
 *
 * @brief Implementation of ssor_sampler.hh
 */

#include "ssor_sampler.hh"

/** apply SSOR sampler */
void SSORSampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const
{
    for (unsigned int k = 0; k < nsmooth; ++k)
    {
        sor_forward.apply(f, x);
        sor_backward.apply(f, x);
    }
}