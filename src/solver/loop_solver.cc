#include "loop_solver.hh"

/** @file loop_solver.hh
 *
 * @brief implementation of loop_solver.hh
 */

/*  Solve the linear system Ax = b */
void LoopSolver::apply(const Eigen::VectorXd &b, Eigen::VectorXd &x)
{
    double r0_nrm = b.norm();
    if (params.verbose >= 2)
    {
        printf("Initial residual ||r_0|| =  %12.4f\n", r0_nrm);
    }
    unsigned int ndof = linear_operator->get_lattice()->M;
    x.setZero();
    bool converged = false;
    double r_nrm;
    double rold_nrm = r0_nrm;
    int niter;
    if (params.verbose >= 2)
        printf("%5s   %8s   %12s   %6s\n", "iter", "||r||", "||r||/||r_0||", "rho");
    for (int k = 0; k < params.maxiter; ++k)
    {
        linear_operator->apply(x, r);
        r -= b;
        r_nrm = r.norm();
        if (params.verbose >= 2)
        {
            printf("%5d   %8.3e   %12.3e   %6.3f\n", k, r_nrm, r_nrm / r0_nrm, r_nrm / rold_nrm);
        }
        if ((r_nrm / r0_nrm < params.rtol) and (r_nrm < params.atol))
        {
            niter = k;
            converged = true;
            break;
        }
        rold_nrm = r_nrm;
        preconditioner->apply(r, Pr);
        x -= Pr;
    }
    if (params.verbose >= 1)
    {
        if (converged)
        {
            printf("Solver converged after %5d iterations\n||r|| = %8.3e, ||r||/||r_0|| = %8.3e\n", niter, r_nrm, r_nrm / r0_nrm);
        }
        else
        {
            printf("Solver failed to converge after %5d iterations\n", params.maxiter);
        }
    }
}