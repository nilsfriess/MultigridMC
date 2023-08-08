#ifndef TEST_SMOOTHER_HH
#define TEST_SMOOTHER_HH TEST_SMOOTHER_HH

#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "lattice/lattice.hh"
#include "smoother/ssor_smoother.hh"
#include "linear_operator/shiftedlaplace_fem_operator.hh"
#include "linear_operator/measured_operator.hh"

/** @brief fixture class for smoother tests */
class SmootherTest : public ::testing::Test
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
        PeriodicCorrelationLengthModelParameters correlationlengthmodel_params;
        correlationlengthmodel_params.kappa_min = 1.2;
        correlationlengthmodel_params.kappa_max = 2.3;
        std::shared_ptr<CorrelationLengthModel> correlationlengthmodel = std::make_shared<PeriodicCorrelationLengthModel>(correlationlengthmodel_params);

        linear_operator = std::make_shared<ShiftedLaplaceFEMOperator>(lattice, correlationlengthmodel);
        unsigned int n_meas = 10;
        std::vector<Eigen::VectorXd> measurement_locations(n_meas);
        Eigen::MatrixXd Sigma(n_meas, n_meas);
        Sigma.setZero();
        for (int k = 0; k < n_meas; ++k)
        {
            measurement_locations[k] = Eigen::Vector2d({uniform_dist(rng), uniform_dist(rng)});
            Sigma(k, k) = 1.E-6 * (1.0 + 2.0 * uniform_dist(rng));
        }
        // Rotate randomly
        Eigen::MatrixXd A(Eigen::MatrixXd::Random(n_meas, n_meas)), Q;
        A.setRandom();
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
        Q = qr.householderQ();
        Sigma = Q * Sigma * Q.transpose();
        std::shared_ptr<ShiftedLaplaceFEMOperator> prior_operator = std::make_shared<ShiftedLaplaceFEMOperator>(lattice,
                                                                                                                correlationlengthmodel);
        MeasurementParameters measurement_params;
        measurement_params.n = n_meas;
        measurement_params.measurement_locations = measurement_locations;
        measurement_params.covariance = Sigma;
        measurement_params.radius = 0.05;
        measurement_params.ignore_measurement_cross_correlations = false;
        measurement_params.measure_global = false;
        measurement_params.sigma_global = 0.0;
        measurement_params.mean_global = 0.0;
        linear_operator_lowrank = std::make_shared<MeasuredOperator>(prior_operator,
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
    std::shared_ptr<ShiftedLaplaceFEMOperator> linear_operator;
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

/* Test standard SSOR smoother
 *
 * Check that the smoother leaves the exact solution invariant
 */
TEST_F(SmootherTest, TestSSORSmoother)
{
    x = x_exact;
    const double omega = 0.8;
    SSORSmoother smoother(linear_operator, omega);
    smoother.apply(b, x);
    double tolerance = 1.E-12;
    double error = (x - x_exact).norm() / x_exact.norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test SSOR smoother with low-rank update
 *
 * Check that the smoother leaves the exact solution unchanged
 */
TEST_F(SmootherTest, TestSSORLowRankSmoother)
{
    x = x_exact;
    const double omega = 0.8;
    SSORSmoother smoother(linear_operator_lowrank, omega);
    smoother.apply(b_lowrank, x);
    double tolerance = 1.E-12;
    double error = (x - x_exact).norm() / x_exact.norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_SMOOTHER_HH