#ifndef TEST_LINEAROPERATOR_HH
#define TEST_LINEAROPERATOR_HH TEST_LINEAROPERATOR_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include "lattice/lattice2d.hh"
#include "lattice/lattice3d.hh"
#include "linear_operator/diffusion_operator.hh"
#include "linear_operator/shiftedlaplace_operator.hh"
#include "linear_operator/shiftedbiharmonic_operator.hh"

/** @brief fixture class for solver tests */
class LinearOperatorTest : public ::testing::Test
{
public:
    LinearOperatorTest() : alpha_K(1.4), alpha_b(3.2){};

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
     * @param[out] u_exact exact solution
     * @param[out] rhs right hand side
     */
    void construct_exact_solution_rhs_shiftedlaplace(std::shared_ptr<Lattice> lattice,
                                                     Eigen::VectorXd &u_exact,
                                                     Eigen::VectorXd &rhs)
    {
        const int dim = lattice->dim();
        double volume = lattice->cell_volume();
        Eigen::VectorXi shape = lattice->shape();
        Eigen::VectorXd h_lat(dim);
        Eigen::VectorXd ones(dim);
        for (int d = 0; d < dim; ++d)
        {
            h_lat[d] = 1 / double(shape[d]);
            ones[d] = 1.0;
        }
        for (unsigned int ell = 0; ell < lattice->Nvertex; ++ell)
        {
            Eigen::VectorXi coord = lattice->vertexidx_linear2euclidean(ell);
            Eigen::VectorXd x = h_lat.cwiseProduct(coord.cast<double>());
            u_exact[ell] = 1.0;
            for (int d = 0; d < dim; ++d)
            {
                u_exact[ell] *= f(x[d]);
            }
            rhs[ell] = alpha_b * u_exact[ell];
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
                rhs[ell] -= alpha_K * dd_u;
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

    /* @brief Construct exact solution for the biharmonic problem
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
     * @param[in] alpha_K coefficient of second order coefficient
     * @param[in] alpha_b coefficient of zero order coefficient
     * @param[out] u_exact exact solution
     * @param[out] rhs right hand side
     */
    void construct_exact_solution_rhs_shiftedbiharmonic(std::shared_ptr<Lattice> lattice,
                                                        Eigen::VectorXd &u_exact,
                                                        Eigen::VectorXd &rhs)
    {
        double volume = lattice->cell_volume();
        Eigen::VectorXi shape = lattice->shape();
        Eigen::VectorXd h_lat(2);
        h_lat[0] = 1 / double(shape[0]);
        h_lat[1] = 1 / double(shape[1]);
        for (unsigned int ell = 0; ell < lattice->Nvertex; ++ell)
        {
            Eigen::VectorXi coord = lattice->vertexidx_linear2euclidean(ell);
            Eigen::VectorXd x = h_lat.cwiseProduct(coord.cast<double>());
            u_exact[ell] = g(x[0]) * g(x[1]);
            rhs[ell] = (alpha_K * alpha_K * (d4_g(x[0]) * g(x[1]) + 2 * d2_g(x[0]) * d2_g(x[1]) + g(x[0]) * d4_g(x[1])) - 2 * alpha_K * alpha_b * (d2_g(x[0]) * g(x[1]) + g(x[0]) * d2_g(x[1])) + alpha_b * alpha_b * u_exact[ell]) * volume;
        }
    }

    /** @brief Coefficient alpha_K */
    const double alpha_K;
    /** @brief Coefficient alpha_b */
    const double alpha_b;
    /** @brief 2d lattice */
    std::shared_ptr<Lattice2d> lattice_2d;
    /** @brief 3d lattice */
    std::shared_ptr<Lattice3d> lattice_3d;
};

/* Test 2d diffusion operator */
TEST_F(LinearOperatorTest, TestDiffusionOperator2d)
{
    double V_cell = lattice_2d->cell_volume();
    std::shared_ptr<DiffusionOperator> diffusion_operator = std::make_shared<DiffusionOperator>(lattice_2d,
                                                                                                alpha_K, 0.0,
                                                                                                alpha_b, 0.0,
                                                                                                0);
    unsigned int nrow = lattice_2d->Nvertex;
    Eigen::VectorXd u_exact(nrow);
    Eigen::VectorXd rhs_exact(nrow);
    Eigen::VectorXd rhs(nrow);
    construct_exact_solution_rhs_shiftedlaplace(lattice_2d, u_exact, rhs_exact);
    diffusion_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / rhs.norm();
    double tolerance = 2.E-4;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test 3d diffusion operator */
TEST_F(LinearOperatorTest, TestDiffusionOperator3d)
{
    double V_cell = lattice_3d->cell_volume();
    std::shared_ptr<DiffusionOperator> diffusion_operator = std::make_shared<DiffusionOperator>(lattice_3d,
                                                                                                alpha_K, 0.0,
                                                                                                alpha_b, 0.0,
                                                                                                0);
    unsigned int nrow = lattice_3d->Nvertex;
    Eigen::VectorXd u_exact(nrow);
    Eigen::VectorXd rhs_exact(nrow);
    Eigen::VectorXd rhs(nrow);
    construct_exact_solution_rhs_shiftedlaplace(lattice_3d, u_exact, rhs_exact);
    diffusion_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / rhs.norm();
    double tolerance = 7.E-3;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test 2d shifted Laplace operator */
TEST_F(LinearOperatorTest, TestShiftedLaplaceOperator2d)
{
    double V_cell = lattice_2d->cell_volume();
    std::shared_ptr<ShiftedLaplaceOperator> shiftedlaplace_operator = std::make_shared<ShiftedLaplaceOperator>(lattice_2d,
                                                                                                               alpha_K,
                                                                                                               alpha_b,
                                                                                                               0);
    unsigned int nrow = lattice_2d->Nvertex;
    Eigen::VectorXd u_exact(nrow);
    Eigen::VectorXd rhs_exact(nrow);
    Eigen::VectorXd rhs(nrow);
    construct_exact_solution_rhs_shiftedlaplace(lattice_2d, u_exact, rhs_exact);
    shiftedlaplace_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / rhs.norm();
    double tolerance = 2.E-4;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test 3d shifted Laplace operator */
TEST_F(LinearOperatorTest, TestShiftedLaplaceOperator3d)
{
    double V_cell = lattice_3d->cell_volume();
    std::shared_ptr<ShiftedLaplaceOperator> shiftedlaplace_operator = std::make_shared<ShiftedLaplaceOperator>(lattice_3d,
                                                                                                               alpha_K, alpha_b,
                                                                                                               0);
    unsigned int nrow = lattice_3d->Nvertex;
    Eigen::VectorXd u_exact(nrow);
    Eigen::VectorXd rhs_exact(nrow);
    Eigen::VectorXd rhs(nrow);
    construct_exact_solution_rhs_shiftedlaplace(lattice_3d, u_exact, rhs_exact);
    shiftedlaplace_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / rhs.norm();
    double tolerance = 7.E-3;
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test 2d shifted Biharmonic operator */
TEST_F(LinearOperatorTest, TestShiftedBiharmonicOperator2d)
{
    double V_cell = lattice_2d->cell_volume();
    std::shared_ptr<ShiftedBiharmonicOperator> shiftedbiharmonic_operator = std::make_shared<ShiftedBiharmonicOperator>(lattice_2d,
                                                                                                                        alpha_K,
                                                                                                                        alpha_b,
                                                                                                                        0);
    unsigned int nrow = lattice_2d->Nvertex;
    Eigen::VectorXd u_exact(nrow);
    Eigen::VectorXd rhs_exact(nrow);
    Eigen::VectorXd rhs(nrow);
    construct_exact_solution_rhs_shiftedbiharmonic(lattice_2d, u_exact, rhs_exact);
    shiftedbiharmonic_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / (rhs).norm();
    double tolerance = 2.5E-2;
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_LINEAROPERATOR_HH
