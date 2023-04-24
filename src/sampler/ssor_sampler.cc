/** @file ssor_sampler.cc
 *
 * @brief Implementation of ssor_sampler.hh
 */

#include "ssor_sampler.hh"

/* apply Sampler */
void SORSampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const
{
    // Diagonal part
    for (unsigned int ell = 0; ell < c_rhs.size(); ++ell)
    {
        double tmp = sqrt_precision_diag[ell];
        c_rhs[ell] = tmp * normal_dist(rng) + f[ell];
    }
    // low-rank correction to covariance matrix
    if (linear_operator->get_m_lowrank() > 0)
    {
        const LinearOperator::DenseMatrixType B = linear_operator->get_B();
        for (unsigned int ell = 0; ell < xi.size(); ++ell)
        {
            xi[ell] = normal_dist(rng);
        }
        c_rhs += B * (*U_lowrank) * xi;
    }
    smoother->apply(c_rhs, x);
}

/** apply SSOR sampler */
void SSORSampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const
{
    sor_forward.apply(f, x);
    sor_backward.apply(f, x);
}