#ifndef TEST_SAMPLER_HH
#define TEST_SAMPLER_HH TEST_SAMPLER_HH

#include <utility>
#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "lattice.hh"
#include "linear_operator.hh"
#include "sampler.hh"

/** @class TestOperator1d
 *
 * @brief Simple linear operator used for testing
 *
 * The sparse matrix is an n x n matrix given by
 *
 *
 *            [  6 -1                 ]
 *            [ -1  6 -1              ]
 *            [    -1  6 -1           ]
 * A_sparse = [       -1  6 -1        ]
 *            [            ....       ]
 *            [               -1 6 -1 ]
 *            [                 -1  6 ]
 *
 * The low rank update is optional. If it is used, the B that defines it is given by
 *
 * B^T = [ 0 0 0 1 0 0 ... 0 ]
 *       [ 0 0 0 0 1 0 ... 0 ]
 *
 *
 * with Sigma = [4.2  -1.1]
 *              [-1.1  9.3]
 */
class TestOperator1d : public LinearOperator
{
public:
    /** @brief Create new instance
     *
     * @param[in] lowrank_update_ Use a low-rank update?
     */
    TestOperator1d(const bool lowrank_update_) : LinearOperator(std::make_shared<Lattice1d>(8),
                                                                lowrank_update_ ? 2 : 0),
                                                 lowrank_update(lowrank_update_)
    {
        typedef Eigen::Triplet<double> T;
        std::vector<T> triplet_list;
        const unsigned int nrow = lattice->M;
        triplet_list.reserve(3 * nrow);
        for (unsigned int i = 0; i < nrow; ++i)
        {
            triplet_list.push_back(T(i, i, +6.0));
            if (i > 0)
            {
                triplet_list.push_back(T(i, (i - 1 + nrow) % nrow, -1.0));
            }
            if (i < nrow - 1)
            {
                triplet_list.push_back(T(i, (i + 1 + nrow) % nrow, -1.0));
            }
        }
        A_sparse.setFromTriplets(triplet_list.begin(), triplet_list.end());
        B = DenseMatrixType(nrow, 2);
        B.setZero();
        B(3, 0) = 10.0;
        B(4, 1) = 10.0;
        DenseMatrixType Sigma(2, 2);
        Sigma << 4.2, -1.1, -1.1, 9.3;
        Sigma_inv = Sigma.inverse();
    }

    /** @brief compute (dense) precision matrix */
    DenseMatrixType precision() const
    {
        DenseMatrixType Q = A_sparse.toDense();
        if (lowrank_update)
        {
            Q += B * Sigma_inv * B.transpose();
        }
        return Q;
    }

    /** @brief compute (dense) covariance matrix */
    DenseMatrixType covariance() const
    {
        return precision().inverse();
    }

protected:
    /** @brief use a low-rank update? */
    const bool lowrank_update;
};

/** @brief fixture class for sampler tests */
class SamplerTest : public ::testing::Test
{
protected:
    /** @brief initialise tests */
    void SetUp() override
    {
    }

    /** @brief Compare measured covariance to true covariance
     *
     * For a randomly chosen f, apply given sampler repeatedly and measure mean
     * and covariance of samples. Returns a tuple, which contains with following
     * quantities:
     *
     *   first:  difference between Q^{-1} * f and sample_mean in the L-infinity norm
     *   second: difference between exact and measured covariance matrix in the
     *           L-infinity norm
     *
     * @param[in] linear_operator Underlying linear operator
     * @param[in] sampler Sampler to be used
     */
    std::pair<double, double> mean_covariance_error(const std::shared_ptr<TestOperator1d> linear_operator,
                                                    const std::shared_ptr<Sampler> sampler)
    {
        unsigned int ndof = linear_operator->get_ndof();
        Eigen::VectorXd f(ndof);
        // Pick an random right hand side f
        unsigned int seed = 1342517;
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> uniform_dist(0, 1);
        Eigen::VectorXd mean_exact(ndof);
        for (unsigned int ell = 0; ell < ndof; ++ell)
        {
            mean_exact[ell] = uniform_dist(rng);
        }
        LinearOperator::DenseMatrixType precision = linear_operator->precision();
        f = precision * mean_exact;
        // Estimators for E(x) and E(x^3)
        Eigen::VectorXd Ex(ndof);
        LinearOperator::DenseMatrixType Exx(ndof, ndof);
        Ex.setZero();
        Exx.setZero();
        unsigned int nsamples_warmup = 1000;
        unsigned int nsamples = 500000;
        Eigen::VectorXd x(ndof);
        for (int k = 0; k < nsamples_warmup; ++k)
        {
            sampler->apply(f, x);
        }
        for (int k = 0; k < nsamples; ++k)
        {
            sampler->apply(f, x);
            Ex += 1. / (k + 1) * (x - Ex);
            Exx += 1. / (k + 1) * (x * x.transpose() - Exx);
        }
        // maximum likelihood estimator
        LinearOperator::DenseMatrixType covariance = Exx - Ex * Ex.transpose();
        LinearOperator::DenseMatrixType covariance_exact = linear_operator->covariance();
        double error_mean = (Ex - mean_exact).lpNorm<Eigen::Infinity>();
        double error_covariance = (covariance - covariance_exact).lpNorm<Eigen::Infinity>();
        return std::make_pair(error_mean, error_covariance);
    }

protected:
};

/* Test Cholesky sampler without low rank correction
 *
 * Draw a large number of samples and check that their covariance agrees with
 * the analytical value of the covariance.
 */
TEST_F(SamplerTest, TestCholeskySampler)
{
    std::shared_ptr<TestOperator1d> linear_operator = std::make_shared<TestOperator1d>(false);
    std::mt19937_64 rng(31841287);
    std::shared_ptr<CholeskySampler> sampler = std::make_shared<CholeskySampler>(linear_operator, rng);
    std::pair<double, double> error = mean_covariance_error(linear_operator, sampler);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error.first, 0.0, tolerance);
    EXPECT_NEAR(error.second, 0.0, tolerance);
}

/* Test SOR sampler without low rank correction
 *
 * Draw a large number of samples and check that their covariance agrees with
 * the analytical value of the covariance.
 */
TEST_F(SamplerTest, TestSORSampler)
{
    std::shared_ptr<TestOperator1d> linear_operator = std::make_shared<TestOperator1d>(false);
    std::mt19937_64 rng(31841287);
    const double omega = 0.8;
    std::shared_ptr<SORSampler> sampler = std::make_shared<SORSampler>(linear_operator,
                                                                       rng,
                                                                       omega,
                                                                       forward);
    std::pair<double, double> error = mean_covariance_error(linear_operator, sampler);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error.first, 0.0, tolerance);
    EXPECT_NEAR(error.second, 0.0, tolerance);
}

/* Test SOR sampler with low rank correction
 *
 * Draw a large number of samples and check that their covariance agrees with
 * the analytical value of the covariance.
 */
TEST_F(SamplerTest, TestSORSamplerLowRank)
{
    std::shared_ptr<TestOperator1d> linear_operator = std::make_shared<TestOperator1d>(true);
    std::mt19937_64 rng(31841287);
    const double omega = 0.8;
    std::shared_ptr<SORSampler> sampler = std::make_shared<SORSampler>(linear_operator,
                                                                       rng,
                                                                       omega,
                                                                       forward);
    std::pair<double, double> error = mean_covariance_error(linear_operator, sampler);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error.first, 0.0, tolerance);
    EXPECT_NEAR(error.second, 0.0, tolerance);
}

/* Test Cholesky sampler with low rank correction
 *
 * Draw a large number of samples and check that their covariance agrees with
 * the analytical value of the covariance.
 */
TEST_F(SamplerTest, TestCholeskySamplerLowRank)
{
    std::shared_ptr<TestOperator1d> linear_operator = std::make_shared<TestOperator1d>(true);
    std::mt19937_64 rng(31841287);
    std::shared_ptr<CholeskySampler> sampler = std::make_shared<CholeskySampler>(linear_operator, rng);
    std::pair<double, double> error = mean_covariance_error(linear_operator, sampler);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error.first, 0.0, tolerance);
    EXPECT_NEAR(error.second, 0.0, tolerance);
}

#endif // TEST_SAMPLER_HH