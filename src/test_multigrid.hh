#ifndef TEST_MULTIGRID_HH
#define TEST_MULTIGRID_HH TEST_MULTIGRID_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include "lattice.hh"
#include "samplestate.hh"
#include "preconditioner.hh"
#include "multigrid_preconditioner.hh"
#include "loop_solver.hh"
#include "diffusion_operator_2d.hh"

/* Test Multigrid solver
 *
 * Computes b = A.x_{exact} for the diffusion operator and then solves A.x = b for x.
 * We expect that ||x-x_{exact}|| is close to zero.
 */
TEST(MultigridTest, TestSolveDiffusion)
{
    unsigned int nx = 32;
    unsigned int ny = 32;
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(nx, ny);
    unsigned int ndof = lattice->M;
    double alpha_K = 1.5;
    double beta_K = 0.3;
    double alpha_b = 1.2;
    double beta_b = 0.1;
    std::shared_ptr<DiffusionOperator2d> lin_op = std::make_shared<DiffusionOperator2d>(lattice,
                                                                                        alpha_K,
                                                                                        beta_K,
                                                                                        alpha_b,
                                                                                        beta_b);
    const unsigned int nlevel = 6;
    const double omega = 0.8;
    std::shared_ptr<SSORSmootherFactory> smoother_factory = std::make_shared<SSORSmootherFactory>(omega);
    std::shared_ptr<IntergridOperator2dLinearFactory> intergrid_operator_factory = std::make_shared<IntergridOperator2dLinearFactory>();
    std::shared_ptr<MultigridPreconditioner> prec = std::make_shared<MultigridPreconditioner>(lin_op,
                                                                                              nlevel,
                                                                                              smoother_factory,
                                                                                              intergrid_operator_factory);
    LoopSolver solver(lin_op, prec, 1.0E-11, 1.0E-9, 100, 0);
    // Create states
    unsigned int seed = 1212417;
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::shared_ptr<SampleState> x_exact = std::make_shared<SampleState>(ndof);
    std::shared_ptr<SampleState> x = std::make_shared<SampleState>(ndof);
    for (unsigned int ell = 0; ell < ndof; ++ell)
    {
        x_exact->data[ell] = dist(rng);
    }
    std::shared_ptr<SampleState> b = std::make_shared<SampleState>(ndof);
    lin_op->apply(x_exact, b);
    solver.apply(b, x);
    double tolerance = 1.E-9;
    double error = (x->data - x_exact->data).norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_MULTIGRID_HH