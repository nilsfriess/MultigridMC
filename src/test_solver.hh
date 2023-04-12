#ifndef TEST_SOLVER_HH
#define TEST_SOLVER_HH TEST_SOLVER_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include "lattice.hh"
#include "samplestate.hh"
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
        // Create states
        unsigned int seed = 1212417;
        std::mt19937 rng(seed);
        std::normal_distribution<double> dist(0.0, 1.0);
        x_exact = std::make_shared<SampleState>(ndof);
        x = std::make_shared<SampleState>(ndof);
        for (unsigned int ell = 0; ell < ndof; ++ell)
        {
            x_exact->data[ell] = dist(rng);
        }

        b = std::make_shared<SampleState>(ndof);
        linear_operator->apply(x_exact, b);
    }

protected:
    /** @brief linear operator */
    std::shared_ptr<DiffusionOperator2d> linear_operator;
    /** @brief exact solution */
    std::shared_ptr<SampleState> x_exact;
    /** @brief numerical solution */
    std::shared_ptr<SampleState> x;
    /** @brief right hand side */
    std::shared_ptr<SampleState> b;
};

/* Test Cholesky solver
 *
 * Computes b = A.x_{exact} for the diffusion operator and then solves A.x = b for x.
 * We expect that ||x-x_{exact}|| is close to zero.
 */
TEST_F(SolverTest, TestCholesky)
{

    CholeskySolver solver(linear_operator);
    solver.apply(b, x);
    double error = (x->data - x_exact->data).norm();
    double tolerance = 1.E-12;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test Multigrid solver
 *
 * Computes b = A.x_{exact} for the diffusion operator and then solves A.x = b for x.
 * We expect that ||x-x_{exact}|| is close to zero.
 */
TEST_F(SolverTest, TestMultigrid)
{
    const double omega = 0.8;
    MultigridParameters multigrid_params;
    multigrid_params.nlevel = 6;
    multigrid_params.npresmooth = 1;
    multigrid_params.npostsmooth = 1;
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
    solver_params.maxiter = 100;
    solver_params.verbose = 0;
    LoopSolver solver(linear_operator, prec, solver_params);
    solver.apply(b, x);
    double tolerance = 1.E-9;
    double error = (x->data - x_exact->data).norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_SOLVER_HH