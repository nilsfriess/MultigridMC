#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Dense>

#include "config.h"
#include "lattice/lattice2d.hh"
#include "smoother/ssor_smoother.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/diffusion_operator_2d.hh"
#include "intergrid/intergrid_operator_2dlinear.hh"
#include "solver/linear_solver.hh"
#include "solver/loop_solver.hh"
#include "solver/cholesky_solver.hh"
#include "preconditioner/multigrid_preconditioner.hh"
#include "auxilliary/parameters.hh"
#include "auxilliary/vtk_writer2d.hh"

/* *********************************************************************** *
 *                                M A I N
 * *********************************************************************** */
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " CONFIGURATIONFILE" << std::endl;
        exit(-1);
    }
    std::string filename(argv[1]);
    std::cout << "Reading parameters from file \'" << filename << "\'" << std::endl;
    GeneralParameters general_params;
    LatticeParameters lattice_params;
    SmootherParameters smoother_params;
    IterativeSolverParameters iterative_solver_params;
    MultigridParameters multigrid_params;
    MeasurementParameters measurement_params;
    general_params.read_from_file(filename);
    lattice_params.read_from_file(filename);
    smoother_params.read_from_file(filename);
    multigrid_params.read_from_file(filename);
    iterative_solver_params.read_from_file(filename);
    measurement_params.read_from_file(filename);

    // Construct lattice and linear operator
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(lattice_params.nx,
                                                                     lattice_params.ny);
    double alpha_K = 1.5;
    double beta_K = 0.3;
    double alpha_b = 1.2;
    double beta_b = 0.1;
    std::shared_ptr<MeasuredDiffusionOperator2d> linear_operator = std::make_shared<MeasuredDiffusionOperator2d>(lattice,
                                                                                                                 measurement_params.measurement_locations,
                                                                                                                 measurement_params.covariance,
                                                                                                                 measurement_params.ignore_measurement_cross_correlations,
                                                                                                                 measurement_params.measure_global,
                                                                                                                 measurement_params.sigma_global,
                                                                                                                 alpha_K,
                                                                                                                 beta_K,
                                                                                                                 alpha_b,
                                                                                                                 beta_b);
    //   Construct smoothers
    /* prepare measurements */
    std::shared_ptr<SmootherFactory> presmoother_factory = std::make_shared<SORSmootherFactory>(smoother_params.omega,
                                                                                                forward);
    std::shared_ptr<SmootherFactory> postsmoother_factory = std::make_shared<SORSmootherFactory>(smoother_params.omega,
                                                                                                 backward);
    std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory = std::make_shared<IntergridOperator2dLinearFactory>();
    std::shared_ptr<LinearSolverFactory> coarse_solver_factory = std::make_shared<CholeskySolverFactory>();
    std::shared_ptr<Preconditioner> multigrid_preconditioner = std::make_shared<MultigridPreconditioner>(linear_operator,
                                                                                                         multigrid_params,
                                                                                                         presmoother_factory,
                                                                                                         postsmoother_factory,
                                                                                                         intergrid_operator_factory,
                                                                                                         coarse_solver_factory);
    std::cout << std::endl;
    // Run sampling experiments
    LoopSolver solver(linear_operator,
                      multigrid_preconditioner,
                      iterative_solver_params);
    // Create states
    unsigned int ndof = linear_operator->get_ndof();
    Eigen::VectorXd x_exact(ndof);
    Eigen::VectorXd x(ndof);
    unsigned int seed = 1482817;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    for (unsigned int ell = 0; ell < ndof; ++ell)
    {
        x_exact[ell] = normal_dist(rng);
    }

    Eigen::VectorXd b(ndof);
    linear_operator->apply(x_exact, b);
    solver.apply(b, x);
    VTKWriter2d vtk_writer("solution.vtk", Cells, lattice, 1);
    vtk_writer.add_state(x_exact, "exact");
    vtk_writer.add_state(x, "numerical");
    vtk_writer.add_state(x - x_exact, "error");
    vtk_writer.write();
}
