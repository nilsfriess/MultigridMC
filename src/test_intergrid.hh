#include <gtest/gtest.h>
#include <random>
#include "intergrid_operator.hh"
#include "diffusion_operator_2d.hh"

/** @brief check that prolongating then restricting will return the same field */
TEST(Intergrid, TestProlongRestrict)
{
    unsigned int n = 32;
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(n, n);
    IntergridOperator2dAvg intergrid_operator_avg(lattice);
    std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();
    unsigned int ndof = lattice->M;
    unsigned int ndof_coarse = coarse_lattice->M;
    std::shared_ptr<SampleState> X = std::make_shared<SampleState>(ndof);
    std::shared_ptr<SampleState> X_coarse = std::make_shared<SampleState>(ndof_coarse);
    std::shared_ptr<SampleState> X_prol_restr = std::make_shared<SampleState>(ndof_coarse);
    unsigned int seed = 1212417;
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    for (unsigned int ell = 0; ell < ndof_coarse; ++ell)
    {
        X_coarse->data[ell] = dist(rng);
    }
    for (unsigned int ell = 0; ell < ndof; ++ell)
    {
        X->data[ell] = 0.0;
    }
    intergrid_operator_avg.prolongate_add(X_coarse, X);
    intergrid_operator_avg.restrict(X, X_prol_restr);
    double tolerance = 1.E-12;
    EXPECT_NEAR((X_prol_restr->data - 4. * X_coarse->data).norm(), 0.0, tolerance);
}

/** @brief check that coarsening the operator works */
TEST(Intergrid, TestCoarsen)
{
    unsigned int n = 32;
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(n, n);
    std::shared_ptr<Lattice2d> coarse_lattice = std::make_shared<Lattice2d>(n / 2, n / 2);
    IntergridOperator2dAvg intergrid_operator_avg(lattice);

    DiffusionOperator2d linear_operator(lattice, 1.0, 0.0, 1.0, 0.0);
    DiffusionOperator2d coarse_operator(coarse_lattice, 1.0, 0.0, 1.0, 0.0);
    LinearOperator coarsened_operator = intergrid_operator_avg.coarsen_operator(linear_operator);
    const double tolerance = 1.E-12;
    EXPECT_NEAR((coarse_operator.as_sparse() - coarsened_operator.as_sparse()).norm(), 0.0, tolerance);
}