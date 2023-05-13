#include <gtest/gtest.h>
#include <memory>
#include <cmath>
#include <random>
#include <Eigen/Dense>
#include "config.h"
#include "statistics.hh"

/** @brief test statistics class
 *
 * The class is tested with an artificial 2d autocorrelated time series of the
 * form
 *
 *   Q_n = A Q_{n-1} + xi_{n-1} + v
 *
 * where A is a contracting 2 x 2 matrix with 0 < rho(A) < 1, xi_j are i.i.d. 2d
 * normal random variables and v is a 2d vector.
 *
 * One can show that for n -> infinity
 *
 *   Q_n = sum_{k=0}^{infinity} A^k (xi_{n-k} + v)
 *
 * and hence
 *
 * Expectation value
 *  E[Q_n] = (I - A)^{-1} v
 *
 * Covariance
 *  Var[Q_n] = E[(Q_n-E[Q_n]) (Q_n-E[Q_n])^T] = (I - A^2)^{-1}
 *
 * Auto-covariance function
 *  C(t) = E[(Q_n - E[Q_n]) (Q_{n+t}) - E[Q_n])^T] = A^t Var[Q_n]
 *
 */
class StatisticsTest : public ::testing::Test
{
public:
    /** @Create a new instance */
    StatisticsTest() : rng(1241517),
                       normal_dist(0, 1) {}

protected:
    /** @brief initialise tests */
    void
    SetUp() override
    {
        // construct iteration matrix
        const double theta = 1.3;
        Eigen::Matrix2d rot;
        rot(0, 0) = cos(theta);
        rot(0, 1) = sin(theta);
        rot(1, 0) = -sin(theta);
        rot(1, 1) = cos(theta);
        Eigen::Matrix2d diag;
        diag.setZero();
        diag(0, 0) = 0.6;
        diag(1, 1) = 0.4;
        A_iter = rot * diag * rot.transpose();
        v_shift(0) = 1.4;
        v_shift(1) = 0.6;
    }

    /** @brief generate samples and recording mean and covariance
     *
     * @param[in] nsamples number of samples to generate
     * @param[inout] stat instance of statistics class
     */

    void generate_samples(const unsigned int nsamples,
                          Statistics &stat)
    {
        const unsigned int nwarmup = 1000000;
        Eigen::Vector2d Q;
        Eigen::Vector2d xi;
        Q.setZero();
        for (unsigned int j = 0; j < nwarmup; ++j)
        {
            xi(0) = normal_dist(rng);
            xi(1) = normal_dist(rng);
            Q = A_iter * Q + xi + v_shift;
        }
        for (unsigned int j = 0; j < nsamples; ++j)
        {
            xi(0) = normal_dist(rng);
            xi(1) = normal_dist(rng);
            Q = A_iter * Q + xi + v_shift;
            stat.record_sample(Q);
        }
    }

    /** @brief random number generator */
    std::mt19937 rng;
    /** @brief normal distribution */
    std::normal_distribution<double> normal_dist;
    /** @brief iteration matrix A*/
    Eigen::Matrix2d A_iter;
    /** @brief shift vector v */
    Eigen::Vector2d v_shift;
};

/** @brief check expectation value */
TEST_F(StatisticsTest, TestAverage)
{
    Statistics stat("test_average", 10);
    const unsigned int nsamples = thorough_testing ? 100000000 : 2000000;
    generate_samples(nsamples, stat);
    Eigen::Vector2d avg_exact = (Eigen::Matrix2d::Identity() - A_iter).inverse() * v_shift;
    Eigen::Vector2d avg_numerical = stat.average();
    const double tolerance = thorough_testing ? 1.E-3 : 3.E-3;
    EXPECT_NEAR((avg_numerical - avg_exact).norm(), 0, tolerance);
}

/** @brief check covariance matrix */
TEST_F(StatisticsTest, TestCovariance)
{
    Statistics stat("test_covariance", 10);
    const unsigned int nsamples = thorough_testing ? 100000000 : 2000000;
    generate_samples(nsamples, stat);
    Eigen::Matrix2d cov_exact = (Eigen::Matrix2d::Identity() - A_iter * A_iter).inverse();
    Eigen::Matrix2d cov_numerical = stat.covariance();
    const double tolerance = thorough_testing ? 1.0E-3 : 3.E-3;
    EXPECT_NEAR((cov_numerical - cov_exact).norm(), 0, tolerance);
}

/** @brief check auto covariance */
TEST_F(StatisticsTest, TestAutoCovariance)
{
    unsigned int window = 6;
    Statistics stat("test_covariance", window);
    const unsigned int nsamples = thorough_testing ? 100000000 : 1000000;
    generate_samples(nsamples, stat);
    Eigen::Matrix2d autocov_exact = (Eigen::Matrix2d::Identity() - A_iter * A_iter).inverse();
    std::vector<Eigen::MatrixXd> autocov_numerical = stat.auto_covariance();
    double diff = 0.0;
    for (int k = 0; k < window; ++k)
    {
        diff += (autocov_numerical[k] - autocov_exact).norm();
        autocov_exact *= A_iter;
    }
    const double tolerance = thorough_testing ? 1.5E-3 : 2.E-2;
    EXPECT_NEAR(diff, 0, tolerance);
}

/** @brief check integrated autocorrelation time */
TEST_F(StatisticsTest, TestIntegratedAutocorrelation)
{
    unsigned int window = 20;
    Statistics stat("test_covariance", window);
    const unsigned int nsamples = thorough_testing ? 100000000 : 1000000;
    generate_samples(nsamples, stat);
    Eigen::Vector2d v;
    v.setZero();
    v(0) = 0.2;
    v(0) = 0.8;
    Eigen::Vector2d w = (Eigen::Matrix2d::Identity() - A_iter * A_iter).inverse() * v;
    Eigen::Vector2d wk = w;
    double tau_int_exact = 1.0;
    for (int k = 1; k < window; ++k)
    {
        wk = A_iter * wk;
        tau_int_exact += 2. * (1. - k / (1.0 * window)) * v.dot(wk) / v.dot(w);
    }
    double tau_int = stat.tau_int(v);
    const double tolerance = thorough_testing ? 1.5E-3 : 2.E-2;
    EXPECT_NEAR(tau_int - tau_int_exact, 0, tolerance);
}