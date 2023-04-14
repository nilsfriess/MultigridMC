#ifndef TEST_SOLVER_HH
#define TEST_SOLVER_HH TEST_SOLVER_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "lattice.hh"
#include "cholesky_solver.hh"
#include "preconditioner.hh"
#include "multigrid_preconditioner.hh"
#include "loop_solver.hh"
#include "diffusion_operator_2d.hh"

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
        unsigned int ndof = lattice->M;
        double alpha_K = 1.5;
        double beta_K = 0.3;
        double alpha_b = 1.2;
        double beta_b = 0.1;

        linear_operator = std::make_shared<DiffusionOperator2d>(lattice,
                                                                alpha_K,
                                                                beta_K,
                                                                alpha_b,
                                                                beta_b);
        unsigned int n_meas = 10;
        std::vector<Eigen::Vector2d> measurement_locations(n_meas);
        Eigen::MatrixXd Sigma(n_meas, n_meas);
        Sigma.setZero();
        for (int k = 0; k < n_meas; ++k)
        {
            measurement_locations[k] = Eigen::Vector2d({uniform_dist(rng), uniform_dist(rng)});
            Sigma(k, k) = 0.001 * (1.0 + 2.0 * uniform_dist(rng));
        }
        // Rotate randomly
        Eigen::MatrixXd A(Eigen::MatrixXd::Random(n_meas, n_meas)), Q;
        A.setRandom();
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
        Q = qr.householderQ();
        Sigma = Q * Sigma * Q.transpose();
        linear_operator_lowrank = std::make_shared<MeasuredDiffusionOperator2d>(lattice,
                                                                                measurement_locations,
                                                                                Sigma,
                                                                                alpha_K,
                                                                                beta_K,
                                                                                alpha_b,
                                                                                beta_b);
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
    std::shared_ptr<DiffusionOperator2d> linear_operator;
    /** @brief linear operator */
    std::shared_ptr<MeasuredDiffusionOperator2d> linear_operator_lowrank;
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
 * We expect that ||x-x_{exact}|| is close to zero.
 */
TEST_F(SolverTest, TestMultigrid)
{
    MultigridParameters multigrid_params;
    multigrid_params.nlevel = 6;
    multigrid_params.npresmooth = 1;
    multigrid_params.npostsmooth = 1;
    const double omega = 1.0;
    std::shared_ptr<SSORSmootherFactory> smoother_factory = std::make_shared<SSORSmootherFactory>(omega);
    std::shared_ptr<IntergridOperator2dLinearFactory> intergrid_operator_factory = std::make_shared<IntergridOperator2dLinearFactory>();
    std::shared_ptr<CholeskySolverFactory> coarse_solver_factory = std::make_shared<CholeskySolverFactory>();
    std::shared_ptr<MultigridPreconditioner> prec = std::make_shared<MultigridPreconditioner>(linear_operator,
                                                                                              multigrid_params,
                                                                                              smoother_factory,
                                                                                              intergrid_operator_factory,
                                                                                              coarse_solver_factory);
    IterativeSolverParameters solver_params;
    solver_params.rtol = 1.0E-11;
    solver_params.atol = 1.0E-9;
    solver_params.maxiter = 20;
    solver_params.verbose = 0;
    LoopSolver solver(linear_operator, prec, solver_params);
    solver.apply(b, x);
    double tolerance = 1.E-9;
    double error = (x - x_exact).norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test Multigrid solver
 *
 * Computes b = A.x_{exact} for the measured diffusion operator and then solves A.x = b for x.
 * We expect that ||x-x_{exact}|| is close to zero.
 */
TEST_F(SolverTest, TestMultigridLowRank)
{
    MultigridParameters multigrid_params;
    multigrid_params.nlevel = 6;
    multigrid_params.npresmooth = 1;
    multigrid_params.npostsmooth = 1;
    const double omega = 1.0;
    std::shared_ptr<SSORLowRankSmootherFactory> smoother_factory = std::make_shared<SSORLowRankSmootherFactory>(omega);
    std::shared_ptr<IntergridOperator2dLinearFactory> intergrid_operator_factory = std::make_shared<IntergridOperator2dLinearFactory>();
    std::shared_ptr<CholeskySolverFactory> coarse_solver_factory = std::make_shared<CholeskySolverFactory>();
    std::shared_ptr<MultigridPreconditioner> prec = std::make_shared<MultigridPreconditioner>(linear_operator_lowrank,
                                                                                              multigrid_params,
                                                                                              smoother_factory,
                                                                                              intergrid_operator_factory,
                                                                                              coarse_solver_factory);
    IterativeSolverParameters solver_params;
    solver_params.rtol = 1.0E-11;
    solver_params.atol = 1.0E-9;
    solver_params.maxiter = 20;
    solver_params.verbose = 0;
    LoopSolver solver(linear_operator_lowrank, prec, solver_params);
    solver.apply(b_lowrank, x);
    double tolerance = 1.E-9;
    double error = (x - x_exact).norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_SOLVER_HH