#ifndef TEST_LINEAROPERATOR_HH
#define TEST_LINEAROPERATOR_HH TEST_LINEAROPERATOR_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include "auxilliary/parameters.hh"
#include "lattice/lattice2d.hh"
#include "lattice/lattice3d.hh"
#include "linear_operator/correlationlength_model.hh"
#include "linear_operator/shiftedlaplace_fem_operator.hh"
#include "linear_operator/shiftedlaplace_fd_operator.hh"
#include "linear_operator/squared_shiftedlaplace_fd_operator.hh"

/** @brief fixture class for solver tests */
class LinearOperatorTest : public ::testing::Test
{

protected:
    /* @brief initialise tests */
    void SetUp() override
    {
        int nx, ny, nz;
        nx = 512;
        ny = 512;
        lattice_2d = std::make_shared<Lattice2d>(nx, ny);
        nx = 64;
        ny = 64;
        nz = 64;
        lattice_3d = std::make_shared<Lattice3d>(nx, ny, nz);
        ConstantCorrelationLengthModelParameters params;
        params.kappa = 2.3;
        correlationlengthmodel = std::make_shared<ConstantCorrelationLengthModel>(params);
    }

protected:
    /** @brief function f(z)
     *
     * @param[in] z value for which the function is to be evaluated
     */
    inline double f(double z)
    {

        return 100 * z * z * (1 - z) * exp(-6 * z);
    }

    /** @brief function f''(z)
     *
     * @param[in] z value for which the function is to be evaluated
     */

    inline double d2_f(double z)
    {
        return 100 * (2 + z * (-30 + z * (72 - 36 * z))) * exp(-6 * z);
    }

    /* @brief Construct exact solution for shifted laplace operator
     *
     * The exact solution is assumed to be of the form
     *
     *      u(x_1,...,x_d) = g(x_1) * g(x_1) * ... * g(x_d)
     *
     * where
     *
     *      g(z) = 100*z^2*(1-x)*exp(-6*z)
     *
     * @param[in] lattice underlying lattice
     * @param[in] correlationlengthmodel model for correlation length
     * @param[out] u_exact exact solution
     * @param[out] rhs right hand side
     */
    void construct_exact_solution_rhs_shiftedlaplace(std::shared_ptr<Lattice> lattice,
                                                     std::shared_ptr<CorrelationLengthModel> correlationlengthmodel,
                                                     Eigen::VectorXd &u_exact,
                                                     Eigen::VectorXd &rhs)
    {
        const int dim = lattice->dim();
        double volume = lattice->cell_volume();
        Eigen::VectorXi shape = lattice->shape();
        for (unsigned int ell = 0; ell < lattice->Nvertex; ++ell)
        {
            Eigen::VectorXd x = lattice->vertex_coordinates(ell);
            u_exact[ell] = 1.0;
            for (int d = 0; d < dim; ++d)
            {
                u_exact[ell] *= f(x[d]);
            }
            rhs[ell] = correlationlengthmodel->kappa_invsq(x) * u_exact[ell];
            for (int j = 0; j < dim; ++j)
            {
                double dd_u = 1.0;
                for (int d = 0; d < dim; ++d)
                {
                    if (j == d)
                        dd_u *= d2_f(x[d]);
                    else
                        dd_u *= f(x[d]);
                }
                rhs[ell] -= dd_u;
            }
            rhs[ell] *= volume;
        }
    }

    /** @brief function g(z)
     *
     * @param[in] z value for which the function is to be evaluated
     */
    inline double g(double z)
    {

        return 2500 * z * z * z * z * (1 - z) * (1 - z) * exp(-8 * z);
    }

    /** @brief second derivative d^2g/dz^2(z)
     *
     * @param[in] z value for which the function is to be evaluated
     */

    inline double d2_g(double z)
    {
        return 5000 * exp(-8 * z) * z * z * (z * (z * (16 * z * (2 * z - 7) + 127) - 52) + 6);
    }

    /** @brief fourth derivative d^4g/dz^4(z)
     *
     * @param[in] z value for which the function is to be evaluated
     */

    inline double d4_g(double z)
    {
        return 20000 * exp(-8 * z) * (z * (z * (32 * z * (z * (16 * (z - 5) * z + 141) - 107) + 1101) - 126) + 3);
    }

    /* @brief Construct exact solution for the squared shifted Laplace problem
     *
     * The exact solution is assumed to be of the form
     *
     *      u(x_1,x_1) = g(x_0) * g(x_1)
     *
     * where
     *
     *      g(z) = 100*z^2*(1-x)*exp(-6*z)
     *
     * @param[in] lattice underlying lattice
     * @param[in] correlationlengthmodel model for correlation length
     * @param[out] u_exact exact solution
     * @param[out] rhs right hand side
     */
    void construct_exact_solution_rhs_squared_shiftedlaplace(std::shared_ptr<Lattice> lattice,
                                                             std::shared_ptr<CorrelationLengthModel> correlationlengthmodel,
                                                             Eigen::VectorXd &u_exact,
                                                             Eigen::VectorXd &rhs)
    {
        double volume = lattice->cell_volume();
        Eigen::VectorXi shape = lattice->shape();
        for (unsigned int ell = 0; ell < lattice->Nvertex; ++ell)
        {
            Eigen::VectorXd x = lattice->vertex_coordinates(ell);
            double alpha_b = correlationlengthmodel->kappa_invsq(x);
            u_exact[ell] = g(x[0]) * g(x[1]);
            rhs[ell] = (d4_g(x[0]) * g(x[1]) + 2 * d2_g(x[0]) * d2_g(x[1]) + g(x[0]) * d4_g(x[1]) - 2 * alpha_b * (d2_g(x[0]) * g(x[1]) + g(x[0]) * d2_g(x[1])) + alpha_b * alpha_b * u_exact[ell]) * volume;
        }
    }

    /** @brief 2d lattice */
    std::shared_ptr<Lattice2d> lattice_2d;
    /** @brief 3d lattice */
    std::shared_ptr<Lattice3d> lattice_3d;
    /** @brief model for correlation length in */
    std::shared_ptr<CorrelationLengthModel> correlationlengthmodel;
};

/* Test 2d FEM shifted Laplace operator */
TEST_F(LinearOperatorTest, TestShiftedLaplaceFEMOperator2d)
{
    double V_cell = lattice_2d->cell_volume();
    std::shared_ptr<ShiftedLaplaceFEMOperator> shiftedlaplace_fem_operator = std::make_shared<ShiftedLaplaceFEMOperator>(lattice_2d,
                                                                                                                         correlationlengthmodel,
                                                                                                                         0);
    unsigned int nrow = lattice_2d->Nvertex;
    Eigen::VectorXd u_exact(nrow);
    Eigen::VectorXd rhs_exact(nrow);
    Eigen::VectorXd rhs(nrow);
    construct_exact_solution_rhs_shiftedlaplace(lattice_2d, correlationlengthmodel, u_exact, rhs_exact);
    shiftedlaplace_fem_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / rhs.norm();
    double tolerance = 2.E-4;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test 3d FEM shifted Laplace operator */
TEST_F(LinearOperatorTest, TestShiftedLaplaceFEMOperator3d)
{
    double V_cell = lattice_3d->cell_volume();
    std::shared_ptr<ShiftedLaplaceFEMOperator> shiftedlaplace_fem_operator = std::make_shared<ShiftedLaplaceFEMOperator>(lattice_3d,
                                                                                                                         correlationlengthmodel,
                                                                                                                         0);
    unsigned int nrow = lattice_3d->Nvertex;
    Eigen::VectorXd u_exact(nrow);
    Eigen::VectorXd rhs_exact(nrow);
    Eigen::VectorXd rhs(nrow);
    construct_exact_solution_rhs_shiftedlaplace(lattice_3d, correlationlengthmodel, u_exact, rhs_exact);
    shiftedlaplace_fem_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / rhs.norm();
    double tolerance = 7.E-3;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test 2d FD shifted Laplace operator */
TEST_F(LinearOperatorTest, TestShiftedLaplaceFDOperator2d)
{
    double V_cell = lattice_2d->cell_volume();
    std::shared_ptr<ShiftedLaplaceFDOperator> shiftedlaplace_fd_operator = std::make_shared<ShiftedLaplaceFDOperator>(lattice_2d,
                                                                                                                      correlationlengthmodel);
    unsigned int nrow = lattice_2d->Nvertex;
    Eigen::VectorXd u_exact(nrow);
    Eigen::VectorXd rhs_exact(nrow);
    Eigen::VectorXd rhs(nrow);
    construct_exact_solution_rhs_shiftedlaplace(lattice_2d, correlationlengthmodel, u_exact, rhs_exact);
    shiftedlaplace_fd_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / rhs.norm();
    double tolerance = 2.E-4;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test 3d FD shifted Laplace operator */
TEST_F(LinearOperatorTest, TestShiftedLaplaceFDOperator3d)
{
    double V_cell = lattice_3d->cell_volume();
    std::shared_ptr<ShiftedLaplaceFDOperator> shiftedlaplace_fd_operator = std::make_shared<ShiftedLaplaceFDOperator>(lattice_3d,
                                                                                                                      correlationlengthmodel,
                                                                                                                      0);
    unsigned int nrow = lattice_3d->Nvertex;
    Eigen::VectorXd u_exact(nrow);
    Eigen::VectorXd rhs_exact(nrow);
    Eigen::VectorXd rhs(nrow);
    construct_exact_solution_rhs_shiftedlaplace(lattice_3d, correlationlengthmodel, u_exact, rhs_exact);
    shiftedlaplace_fd_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / rhs.norm();
    double tolerance = 7.E-3;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test 2d FD squared shifted Laplace operator */
TEST_F(LinearOperatorTest, TestSquaredShiftedLaplaceFDOperator2d)
{
    double V_cell = lattice_2d->cell_volume();
    std::shared_ptr<SquaredShiftedLaplaceFDOperator> squared_shiftedlaplace_fd_operator = std::make_shared<SquaredShiftedLaplaceFDOperator>(lattice_2d,
                                                                                                                                            correlationlengthmodel,
                                                                                                                                            0);
    unsigned int nrow = lattice_2d->Nvertex;
    Eigen::VectorXd u_exact(nrow);
    Eigen::VectorXd rhs_exact(nrow);
    Eigen::VectorXd rhs(nrow);
    construct_exact_solution_rhs_squared_shiftedlaplace(lattice_2d, correlationlengthmodel, u_exact, rhs_exact);
    squared_shiftedlaplace_fd_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / (rhs).norm();
    double tolerance = 2.5E-2;
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_LINEAROPERATOR_HH
