#include "multigridmc_sampler.hh"
/** @file multigridmc_sampler.cc
 *
 * @brief Implementation of multigridmc_sampler.hh
 */

/** Create a new instance */
MultigridMCSampler::MultigridMCSampler(std::shared_ptr<LinearOperator> linear_operator_,
                                       std::shared_ptr<RandomGenerator> rng_,
                                       const MultigridParameters params_,
                                       const CholeskyParameters cholesky_params_) : Sampler(linear_operator_, rng_),
                                                                                    params(params_),
                                                                                    cholesky_params(cholesky_params_)
{
    // Extract underlying fine lattice
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    // Linear operator on a given level
    std::shared_ptr<LinearOperator> lin_op = linear_operator;
    if (params.verbose > 0)
    {
        std::cout << "Setting up Multilevel MC sampler " << std::endl;
    }

    std::shared_ptr<SamplerFactory> presampler_factory;
    std::shared_ptr<SamplerFactory> postsampler_factory;
    if (params.smoother == "SOR")
    {
        presampler_factory = std::make_shared<SORSamplerFactory>(rng,
                                                                 params.omega,
                                                                 params.npresmooth,
                                                                 forward);
        postsampler_factory = std::make_shared<SORSamplerFactory>(rng,
                                                                  params.omega,
                                                                  params.npostsmooth,
                                                                  backward);
    }
    else if (params.smoother == "SSOR")
    {
        presampler_factory = std::make_shared<SSORSamplerFactory>(rng,
                                                                  params.omega,
                                                                  params.npresmooth);
        postsampler_factory = std::make_shared<SSORSamplerFactory>(rng,
                                                                   params.omega,
                                                                   params.npostsmooth);
    }
    else
    {
        std::cout << "ERROR: invalid sampler \'" << params.smoother << "\'" << std::endl;
        exit(-1);
    }
    std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory = std::make_shared<IntergridOperatorLinearFactory>();
    std::shared_ptr<SamplerFactory> coarse_sampler_factory;
    if (params.coarse_solver == "Cholesky")
    {
        if (cholesky_params.factorisation == SparseFactorisation)
        {
            coarse_sampler_factory = std::make_shared<SparseCholeskySamplerFactory>(rng);
        }
        else if (cholesky_params.factorisation == DenseFactorisation)
        {
            coarse_sampler_factory = std::make_shared<DenseCholeskySamplerFactory>(rng);
        }
    }
    else if (params.coarse_solver == "SSOR")
    {
        coarse_sampler_factory = std::make_shared<SSORSamplerFactory>(rng,
                                                                      params.omega,
                                                                      params.ncoarsesmooth);
    }
    else
    {
        std::cout << "ERROR: multigrid coarse sampler \'" << params.coarse_solver << "\'" << std::endl;
        exit(-1);
    }

    for (int level = 0; level < params.nlevel; ++level)
    {
        if (params.verbose > 0)
        {
            std::cout << "  level " << level << " lattice : " << lattice->get_info() << std::endl;
        }
        x_ell.push_back(Eigen::VectorXd(lattice->Nvertex));
        f_ell.push_back(Eigen::VectorXd(lattice->Nvertex));
        r_ell.push_back(Eigen::VectorXd(lattice->Nvertex));
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
        int cycle_ = (level > 0) ? params.cycle : 1;
        for (int j = 0; j < cycle_; ++j)
        {
            // Presampler
            presamplers[level]->apply(f_ell[level], x_ell[level]);
            // Compute residual
            linear_operators[level]->apply(x_ell[level], r_ell[level]);
            r_ell[level] = f_ell[level] - r_ell[level];
            intergrid_operators[level]->restrict(r_ell[level], f_ell[level + 1]);
            // Recursive call
            x_ell[level + 1].setZero();
            sample(level + 1);
            // Prolongate and add
            intergrid_operators[level]->prolongate_add(params.coarse_scaling, x_ell[level + 1], x_ell[level]);
            // Postsmooth
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