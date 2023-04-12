#include "multigrid_preconditioner.hh"
/** @file multigrid_preconditioner.cc
 *
 * @brief Implementation of multigrid_preconditioner.hh
 */

/** Create a new instance */
MultigridPreconditioner::MultigridPreconditioner(std::shared_ptr<LinearOperator> linear_operator_,
                                                 const MultigridParameters params_,
                                                 std::shared_ptr<SmootherFactory> smoother_factory_,
                                                 std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory_,
                                                 std::shared_ptr<LinearSolverFactory> coarse_solver_factory_) : Preconditioner(linear_operator_),
                                                                                                                params(params_),
                                                                                                                smoother_factory(smoother_factory_),
                                                                                                                intergrid_operator_factory(intergrid_operator_factory_),
                                                                                                                coarse_solver_factory(coarse_solver_factory_)
{
    // Extract underlying fine lattice
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    // Linear operator on a given level
    std::shared_ptr<LinearOperator> lin_op = linear_operator;
    for (int level = 0; level < params.nlevel; ++level)
    {
        x_ell.push_back(std::make_shared<SampleState>(lattice->M));
        b_ell.push_back(std::make_shared<SampleState>(lattice->M));
        r_ell.push_back(std::make_shared<SampleState>(lattice->M));
        linear_operators.push_back(lin_op);
        std::shared_ptr<Smoother> smoother = smoother_factory->get(lin_op);
        smoothers.push_back(smoother);
        if (level < params.nlevel - 1)
        {
            std::shared_ptr<IntergridOperator> intergrid_operator = intergrid_operator_factory->get(lattice);
            intergrid_operators.push_back(intergrid_operator);
            lin_op = std::make_shared<LinearOperator>(intergrid_operator->coarsen_operator(*lin_op));
            //  Move to next-coarser lattice
            lattice = lattice->get_coarse_lattice();
        }
    }
    coarse_solver = coarse_solver_factory->get(lin_op);
}

/** Recursive solve on a givel level */
void MultigridPreconditioner::solve(const unsigned int level)
{
    x_ell[level]->data.setZero();
    if (level == params.nlevel - 1)
    {
        // Coarse level solve
        coarse_solver->apply(b_ell[level], x_ell[level]);
    }
    else
    {
        // Presmooth
        for (unsigned int k = 0; k < params.npresmooth; ++k)
        {
            smoothers[level]->apply(b_ell[level], x_ell[level]);
        }
        // Compute residual
        linear_operators[level]->apply(x_ell[level], r_ell[level]);
        r_ell[level]->data = b_ell[level]->data - r_ell[level]->data;
        intergrid_operators[level]->restrict(r_ell[level], b_ell[level + 1]);
        // Recursive call
        solve(level + 1);
        // Prolongate and add
        intergrid_operators[level]->prolongate_add(x_ell[level + 1], x_ell[level]);
        // Postsmooth
        for (unsigned int k = 0; k < params.npostsmooth; ++k)
        {
            smoothers[level]->apply(b_ell[level], x_ell[level]);
        }
    }
}

/** Solve the linear system Ax = b with one iteration of the multigrid V-cycle */
void MultigridPreconditioner::apply(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x)
{
    b_ell[0]->data = b->data;
    for (unsigned int ell = 0; ell < x->data.size(); ++ell)
    {
        x_ell[0]->data[ell] = 0;
    }
    solve(0);
    x->data = x_ell[0]->data;
}