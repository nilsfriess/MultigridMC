#include <iostream>
#include <chrono>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/QR>

#include "lattice/lattice2d.hh"
#include "smoother/smoother.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/diffusion_operator_2d.hh"
#include "intergrid/intergrid_operator.hh"
#include "solver/cholesky_solver.hh"
#include "solver/iterative_solver.hh"
#include "preconditioner/multigrid_preconditioner.hh"
#include "auxilliary/vtk_writer2d.hh"

int main(int argc, char *argv[])
{
    unsigned int nx, ny;
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " NX NY" << std::endl;
        exit(-1);
    }
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    unsigned int seed = 1212417;
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist_normal(0.0, 1.0);
    std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);
    std::cout << "lattice size : " << nx << " x " << ny << std::endl;
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(nx, ny);
    unsigned int ndof = lattice->M;
    double alpha_K = 1.5;
    double beta_K = 0.3;
    double alpha_b = 1.2;
    double beta_b = 0.1;
    unsigned int n_meas = 20;
    std::cout << "Number of measurements : " << n_meas << std::endl;
    std::vector<Eigen::Vector2d> measurement_locations(n_meas);
    Eigen::MatrixXd Sigma(n_meas, n_meas);
    Sigma.setZero();
    for (int k = 0; k < n_meas; ++k)
    {
        measurement_locations[k] = Eigen::Vector2d({dist_uniform(rng), dist_uniform(rng)});
        Sigma(k, k) = 0.0001 * (1.0 + 2.0 * dist_uniform(rng));
    }
    // Rotate randomly
    Eigen::MatrixXd A(Eigen::MatrixXd::Random(n_meas, n_meas)), Q;
    A.setRandom();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Q = qr.householderQ();
    Sigma = Q * Sigma * Q.transpose();
    std::shared_ptr<MeasuredDiffusionOperator2d>
        linear_operator = std::make_shared<MeasuredDiffusionOperator2d>(lattice,
                                                                        measurement_locations,
                                                                        Sigma,
                                                                        alpha_K,
                                                                        beta_K,
                                                                        alpha_b,
                                                                        beta_b);
    Eigen::VectorXd x_exact(ndof);
    Eigen::VectorXd x(ndof);
    for (unsigned int ell = 0; ell < ndof; ++ell)
    {
        x_exact[ell] = dist_normal(rng);
    }

    Eigen::VectorXd b(ndof);
    linear_operator->apply(x_exact, b);
    MultigridParameters multigrid_params;
    multigrid_params.nlevel = 6;
    multigrid_params.npresmooth = 2;
    multigrid_params.npostsmooth = 2;
    const double omega = 1.0;
    std::cout << "omega = " << omega << std::endl;
    std::shared_ptr<SSORSmootherFactory> smoother_factory = std::make_shared<SSORSmootherFactory>(omega);
    std::shared_ptr<IntergridOperator2dLinearFactory> intergrid_operator_factory = std::make_shared<IntergridOperator2dLinearFactory>();
    std::shared_ptr<CholeskySolverFactory> coarse_solver_factory = std::make_shared<CholeskySolverFactory>();
    std::shared_ptr<MultigridPreconditioner> prec = std::make_shared<MultigridPreconditioner>(linear_operator,
                                                                                              multigrid_params,
                                                                                              smoother_factory,
                                                                                              intergrid_operator_factory,
                                                                                              coarse_solver_factory);
    IterativeSolverParameters solver_params;
    solver_params.rtol = 1.0E-12;
    solver_params.atol = 1.0;
    solver_params.maxiter = 100;
    solver_params.verbose = 2;
    LoopSolver solver(linear_operator, prec, solver_params);
    solver.apply(b, x);
    double error = (x - x_exact).norm();
    std::cout << "error ||u-u_{exact}|| = " << error << std::endl;
    VTKWriter2d vtk_writer("output.vtk", Cells, lattice);
    vtk_writer.add_state(x_exact, "exact_solution");
    vtk_writer.add_state(x, "numerical_solution");
    vtk_writer.add_state(x - x_exact, "error");
    vtk_writer.write();
}
