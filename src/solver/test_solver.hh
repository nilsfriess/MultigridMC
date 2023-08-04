#ifndef TEST_SOLVER_HH
#define TEST_SOLVER_HH TEST_SOLVER_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "lattice/lattice2d.hh"
#include "smoother/ssor_smoother.hh"
#include "intergrid/intergrid_operator_linear.hh"
#include "preconditioner/preconditioner.hh"
#include "preconditioner/multigrid_preconditioner.hh"
#include "solver/cholesky_solver.hh"
#include "solver/loop_solver.hh"
#include "linear_operator/diffusion_operator.hh"
#include "linear_operator/measured_operator.hh"

/** @brief fixture class for solver tests */
class SolverTest : public ::testing::Test
{
protected:
    /* @brief initialise tests */
    void SetUp() override
    {
        unsigned int nx = 32;
        unsigned int ny = 32;

        unsigned int seed = 1212417;
        std::mt19937 rng(seed);
        std::normal_distribution<double> normal_dist(0.0, 1.0);
        std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

        std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(nx, ny);
        unsigned int ndof = lattice->Nvertex;
        double alpha_K = 1.5;
        double beta_K = 0.3;
        double alpha_b = 1.2;
        double beta_b = 0.1;

        linear_operator = std::make_shared<DiffusionOperator>(lattice,
                                                              alpha_K,
                                                              beta_K,
                                                              alpha_b,
                                                              beta_b);
        unsigned int n_meas = 10;
        std::vector<Eigen::VectorXd> measurement_locations(n_meas);
        Eigen::MatrixXd Sigma(n_meas, n_meas);
        Sigma.setZero();
        for (int k = 0; k < n_meas; ++k)
        {
            measurement_locations[k] = Eigen::Vector2d({uniform_dist(rng), uniform_dist(rng)});
            Sigma(k, k) = 0.1 * (1.0 + 2.0 * uniform_dist(rng));
        }
        // Rotate randomly
        Eigen::MatrixXd A(Eigen::MatrixXd::Random(n_meas, n_meas)), Q;
        A.setRandom();
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
        Q = qr.householderQ();
        Sigma = Q * Sigma * Q.transpose();
        std::shared_ptr<DiffusionOperator> diffusion_operator = std::make_shared<DiffusionOperator>(lattice,
                                                                                                    alpha_K,
                                                                                                    beta_K,
                                                                                                    alpha_b,
                                                                                                    beta_b);
        MeasurementParameters measurement_params;
        measurement_params.measurement_locations = measurement_locations;
        measurement_params.covariance = Sigma;
        measurement_params.ignore_measurement_cross_correlations = false;
        measurement_params.measure_global = false;
        measurement_params.sigma_global = 0.0;
        measurement_params.mean_global = 0.0;
        linear_operator_lowrank = std::make_shared<MeasuredOperator>(diffusion_operator,
                                                                     measurement_params);
        // Create states
        x_exact = Eigen::VectorXd(ndof);
        x = Eigen::VectorXd(ndof);
        for (unsigned int ell = 0; ell < ndof; ++ell)
        {
            x_exact[ell] = normal_dist(rng);
        }

        b = Eigen::VectorXd(ndof);
        linear_operator->apply(x_exact, b);
        b_lowrank = Eigen::VectorXd(ndof);
        linear_operator_lowrank->apply(x_exact, b_lowrank);
    }

protected:
    /** @brief linear operator */
    std::shared_ptr<DiffusionOperator> linear_operator;
    /** @brief linear operator */
    std::shared_ptr<MeasuredOperator> linear_operator_lowrank;
    /** @brief exact solution */
    Eigen::VectorXd x_exact;
    /** @brief numerical solution */
    Eigen::VectorXd x;
    /** @brief right hand side */
    Eigen::VectorXd b;
    /** @brief right hand side */
    Eigen::VectorXd b_lowrank;
};

/* Test Cholesky solver
 *
 * Computes b = A.x_{exact} for the diffusion operator and then solves A.x = b for x.
 * We expect that ||x-x_{exact}|| is close to zero.
 */
TEST_F(SolverTest, TestCholesky)
{

    CholeskySolver solver(linear_operator_lowrank);
    solver.apply(b_lowrank, x);
    double error = (x - x_exact).norm();
    double tolerance = 1.E-12;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test Multigrid solver for standard diffusion operator
 *
 * Computes b = A.x_{exact} for the diffusion operator and then solves A.x = b for x.
 * We expect that ||x-x_{exact}||/||x-x_{exact}|| is close to zero.
 */
TEST_F(SolverTest, TestMultigrid)
{
    MultigridParameters multigrid_params;
    multigrid_params.nlevel = 5;
    multigrid_params.npresmooth = 1;
    multigrid_params.npostsmooth = 1;
    multigrid_params.cycle = 1;
    const double omega = 1.0;
    std::shared_ptr<SSORSmootherFactory> presmoother_factory = std::make_shared<SSORSmootherFactory>(omega);
    std::shared_ptr<SSORSmootherFactory> postsmoother_factory = std::make_shared<SSORSmootherFactory>(omega);
    std::shared_ptr<IntergridOperatorLinearFactory> intergrid_operator_factory = std::make_shared<IntergridOperatorLinearFactory>();
    std::shared_ptr<CholeskySolverFactory> coarse_solver_factory = std::make_shared<CholeskySolverFactory>();
    std::shared_ptr<MultigridPreconditioner> prec = std::make_shared<MultigridPreconditioner>(linear_operator,
                                                                                              multigrid_params,
                                                                                              presmoother_factory,
                                                                                              postsmoother_factory,
                                                                                              intergrid_operator_factory,
                                                                                              coarse_solver_factory);
    IterativeSolverParameters solver_params;
    solver_params.rtol = 1.0E-13;
    solver_params.atol = 1.0E-12;
    solver_params.maxiter = 100;
    solver_params.verbose = 0;
    LoopSolver solver(linear_operator, prec, solver_params);
    solver.apply(b, x);
    double tolerance = 1.E-10;
    double error = (x - x_exact).norm() / x_exact.norm();
    ;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test Multigrid solver
 *
 * Computes b = A.x_{exact} for the measured diffusion operator and then solves A.x = b for x.
 * We expect that ||x-x_{exact}||/||x-x_{exact}|| is close to zero.
 */
TEST_F(SolverTest, TestMultigridLowRank)
{
    MultigridParameters multigrid_params;
    multigrid_params.nlevel = 5;
    multigrid_params.npresmooth = 1;
    multigrid_params.npostsmooth = 1;
    multigrid_params.cycle = 1;
    const double omega = 1.0;
    std::shared_ptr<SSORSmootherFactory> presmoother_factory = std::make_shared<SSORSmootherFactory>(omega);
    std::shared_ptr<SSORSmootherFactory> postsmoother_factory = std::make_shared<SSORSmootherFactory>(omega);
    std::shared_ptr<IntergridOperatorLinearFactory> intergrid_operator_factory = std::make_shared<IntergridOperatorLinearFactory>();
    std::shared_ptr<CholeskySolverFactory> coarse_solver_factory = std::make_shared<CholeskySolverFactory>();
    std::shared_ptr<MultigridPreconditioner> prec = std::make_shared<MultigridPreconditioner>(linear_operator_lowrank,
                                                                                              multigrid_params,
                                                                                              presmoother_factory,
                                                                                              postsmoother_factory,
                                                                                              intergrid_operator_factory,
                                                                                              coarse_solver_factory);
    IterativeSolverParameters solver_params;
    solver_params.rtol = 1.0E-13;
    solver_params.atol = 1.0E-11;
    solver_params.maxiter = 100;
    solver_params.verbose = 0;
    LoopSolver solver(linear_operator_lowrank, prec, solver_params);
    solver.apply(b_lowrank, x);
    double tolerance = 1.E-10;
    double error = (x - x_exact).norm() / x_exact.norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_SOLVER_HH
