#ifndef TEST_LINEAROPERATOR_HH
#define TEST_LINEAROPERATOR_HH TEST_LINEAROPERATOR_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include "lattice/lattice2d.hh"
#include "linear_operator/diffusion_operator.hh"
#include "linear_operator/shiftedlaplace_operator.hh"

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
        nx = 256;
        ny = 256;
        lattice_2d = std::make_shared<Lattice2d>(nx, ny);
        nx = 64;
        ny = 64;
        nz = 64;
        lattice_3d = std::make_shared<Lattice3d>(nx, ny, nz);
    }

protected:
    /** @brief function g(z)
     *
     * @param[in] z value for which the function is to be evaluated
     */
    inline double g(double z)
    {

        return 100 * z * z * (1 - z) * exp(-6 * z);
    }

    /** @brief function g''(z)
     *
     * @param[in] z value for which the function is to be evaluated
     */

    inline double dd_g(double z)
    {
        return 100 * (2 + z * (-30 + z * (72 - 36 * z))) * exp(-6 * z);
    }

    /* @brief Construct exact solution
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
    void construct_exact_solution_rhs(std::shared_ptr<Lattice> lattice,
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
                u_exact[ell] *= g(x[d]);
            }
            rhs[ell] = alpha_b * u_exact[ell];
            for (int j = 0; j < dim; ++j)
            {
                double dd_u = 1.0;
                for (int d = 0; d < dim; ++d)
                {
                    if (j == d)
                        dd_u *= dd_g(x[d]);
                    else
                        dd_u *= g(x[d]);
                }
                rhs[ell] -= alpha_K * dd_u;
            }
            rhs[ell] *= volume;
        }
    }

    /** @brief Coefficient alpha_K */
    const double alpha_K;
    /** @brief Coefficient alpha_b */
    const double alpha_b;
    /** @brief 2d lattice */
    std::shared_ptr<Lattice> lattice_2d;
    /** @brief 3d lattice */
    std::shared_ptr<Lattice> lattice_3d;
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
    construct_exact_solution_rhs(lattice_2d, u_exact, rhs_exact);
    diffusion_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / sqrt(V_cell);
    double tolerance = 2.E-2;
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
    construct_exact_solution_rhs(lattice_3d, u_exact, rhs_exact);
    diffusion_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / sqrt(V_cell);
    double tolerance = 0.2;
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
    construct_exact_solution_rhs(lattice_2d, u_exact, rhs_exact);
    shiftedlaplace_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / sqrt(V_cell);
    double tolerance = 2.E-2;
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
    construct_exact_solution_rhs(lattice_3d, u_exact, rhs_exact);
    shiftedlaplace_operator->apply(u_exact, rhs);
    double error = (rhs - rhs_exact).norm() / sqrt(V_cell);
    double tolerance = 0.2;
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_LINEAROPERATOR_HH
