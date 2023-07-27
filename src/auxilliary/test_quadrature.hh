#include <gtest/gtest.h>
#include "quadrature.hh"
#include <Eigen/Dense>

/** @brief test quadrature class
 */
class QuadratureTest : public ::testing::Test
{
public:
    /** @Create a new instance */
    QuadratureTest() : quadrature_order_0(3, 0),
                       quadrature_order_1(3, 1),
                       quadrature_order_2(3, 2) {}

protected:
    /** @brief initialise tests */
    void
    SetUp() override
    {
    }

    /** @brief numerically integrate monomial with a given power
     *
     * integrate (alpha+1)*(beta+1)*(gamma+1) x^alpha * y^beta * z^beta in the unit cube [0,1]^3
     * with a given quadrature rule. The exact value of this integral is 1.
     */
    double integrate_monomial(const GaussLegendreQuadrature quadrature,
                              const int alpha,
                              const int beta,
                              const int gamma)
    {
        std::vector<double> weights = quadrature.get_weights();
        std::vector<Eigen::VectorXd> points = quadrature.get_points();
        int npoints = weights.size();
        double integral = 0.0;
        for (int j = 0; j < npoints; ++j)
        {
            Eigen::VectorXd pt = points[j];
            integral += weights[j] * pow(pt[0], alpha) * pow(pt[1], beta) * pow(pt[2], gamma);
        }
        return (alpha + 1.0) * (beta + 1.0) * (gamma + 1.0) * integral;
    }

    /** @brief Gaussian quadrature of order 0 */
    const GaussLegendreQuadrature quadrature_order_0;
    /** @brief Gaussian quadrature of order 1 */
    const GaussLegendreQuadrature quadrature_order_1;
    /** @brief Gaussian quadrature of order 2 */
    const GaussLegendreQuadrature quadrature_order_2;
};

/** @brief Check that quadrature or order 0 can functions that are multi-linear  */
TEST_F(QuadratureTest, TestQuadratureOrder0)
{
    double tolerance = 1.0E-12;
    EXPECT_NEAR(integrate_monomial(quadrature_order_0, 1, 1, 1), 1.0, tolerance);
}

/** @brief Check that quadrature or order 1 can functions that are up to cubic  */
TEST_F(QuadratureTest, TestQuadratureOrder1)
{
    double tolerance = 1.0E-12;
    for (int alpha = 0; alpha < 4; ++alpha)
        for (int beta = 0; beta < 4; ++beta)
            for (int gamma = 0; gamma < 4; ++gamma)
                EXPECT_NEAR(integrate_monomial(quadrature_order_1, alpha, beta, gamma), 1.0, tolerance);
}

/** @brief Check that quadrature or order 1 can functions that are up to deree 5  */
TEST_F(QuadratureTest, TestQuadratureOrder2)
{
    double tolerance = 1.0E-12;
    for (int alpha = 0; alpha < 6; ++alpha)
        for (int beta = 0; beta < 6; ++beta)
            for (int gamma = 0; gamma < 6; ++gamma)
                EXPECT_NEAR(integrate_monomial(quadrature_order_2, alpha, beta, gamma), 1.0, tolerance);
}