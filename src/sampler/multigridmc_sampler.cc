#include "multigridmc_sampler.hh"
/** @file multigridmc_sampler.cc
 *
 * @brief Implementation of multigridmc_sampler.hh
 */

/** Create a new instance */
MultigridMCSampler::MultigridMCSampler(std::shared_ptr<LinearOperator> linear_operator_,
                                       std::mt19937_64 &rng_,
                                       const MultigridMCParameters params_,
                                       std::shared_ptr<SamplerFactory> presampler_factory_,
                                       std::shared_ptr<SamplerFactory> postsampler_factory_,
                                       std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory_,
                                       std::shared_ptr<SamplerFactory> coarse_sampler_factory_) : Sampler(linear_operator_, rng_),
                                                                                                  params(params_),
                                                                                                  presampler_factory(presampler_factory_),
                                                                                                  postsampler_factory(postsampler_factory_),
                                                                                                  intergrid_operator_factory(intergrid_operator_factory_),
                                                                                                  coarse_sampler_factory(coarse_sampler_factory_)
{
    // Extract underlying fine lattice
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    // Linear operator on a given level
    std::shared_ptr<LinearOperator> lin_op = linear_operator;
    if (params.verbose > 0)
    {
        std::cout << "Setting up Multilevel MC sampler " << std::endl;
    }
    for (int level = 0; level < params.nlevel; ++level)
    {
        if (params.verbose > 0)
        {
            std::cout << "  level " << level << " lattice : " << lattice->get_info() << std::endl;
        }
        x_ell.push_back(Eigen::VectorXd(lattice->Ncell));
        f_ell.push_back(Eigen::VectorXd(lattice->Ncell));
        r_ell.push_back(Eigen::VectorXd(lattice->Ncell));
        linear_operators.push_back(lin_op);
        std::shared_ptr<Sampler> presampler = presampler_factory->get(lin_op);
        presamplers.push_back(presampler);
        std::shared_ptr<Sampler> postsampler = postsampler_factory->get(lin_op);
        postsamplers.push_back(postsampler);
        if (level < params.nlevel - 1)
        {
            std::shared_ptr<IntergridOperator> intergrid_operator = intergrid_operator_factory->get(lattice);
            intergrid_operators.push_back(intergrid_operator);
            lin_op = std::make_shared<LinearOperator>(lin_op->coarsen(intergrid_operator));
            //  Move to next-coarser lattice
            lattice = lattice->get_coarse_lattice();
        }
    }
    coarse_sampler = coarse_sampler_factory->get(lin_op);
}

/** Recursive sampling on a given level */
void MultigridMCSampler::sample(const unsigned int level) const
{
    if (level == params.nlevel - 1)
    {
        // Coarse level solve
        coarse_sampler->apply(f_ell[level], x_ell[level]);
    }
    else
    {
        // Presampler
        for (unsigned int k = 0; k < params.npresample; ++k)
        {
            presamplers[level]->apply(f_ell[level], x_ell[level]);
        }
        // Compute residual
        linear_operators[level]->apply(x_ell[level], r_ell[level]);
        r_ell[level] = f_ell[level] - r_ell[level];
        intergrid_operators[level]->restrict(r_ell[level], f_ell[level + 1]);
        // Recursive call
        x_ell[level + 1].setZero();
        sample(level + 1);
        // Prolongate and add
        intergrid_operators[level]->prolongate_add(x_ell[level + 1], x_ell[level]);
        // Postsmooth
        for (unsigned int k = 0; k < params.npostsample; ++k)
        {
            postsamplers[level]->apply(f_ell[level], x_ell[level]);
        }
    }
}

/** Solve the linear system Ax = b with one iteration of the multigrid V-cycle */
void MultigridMCSampler::apply(const Eigen::VectorXd &f, Eigen::VectorXd &x) const
{
    f_ell[0] = f;
    x_ell[0] = x;
    sample(0);
    x = x_ell[0];
}