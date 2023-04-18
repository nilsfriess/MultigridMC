#ifndef TEST_SAMPLER_HH
#define TEST_SAMPLER_HH TEST_SAMPLER_HH

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
        B(3, 0) = 1.0;
        B(4, 1) = 1.0;
        DenseMatrixType Sigma(2, 2);
        Sigma << 4.2, -1.1, -1.1, 9.3;
        Sigma_inv = Sigma.inverse();
    }

    /** @brief compute (dense) covariance matrix */
    DenseMatrixType covariance() const
    {
        DenseMatrixType Q = A_sparse.toDense();
        if (lowrank_update)
        {
            Q += B * Sigma_inv * B.transpose();
        }
        return Q.inverse();
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
     * Apply given smoother repeatedly and measure covariance of samples. Returns
     * the difference between true and measured covariance in L-infinity norm.
     *
     * @param[in] linear_operator Underlying linear operator
     * @param[in] sampler Sampler to be used
     */
    double covariance_error(const std::shared_ptr<TestOperator1d> linear_operator,
                            const std::shared_ptr<Sampler> sampler)
    {
        unsigned int ndof = linear_operator->get_ndof();
        Eigen::VectorXd x(ndof);
        Eigen::VectorXd b(ndof);
        LinearOperator::DenseMatrixType covariance(ndof, ndof);
        covariance.setZero();
        unsigned int nsamples = 100000;
        for (int k = 0; k < nsamples; ++k)
        {
            sampler->apply(b, x);
            covariance += 1. / (k + 1) * (x * x.transpose() - covariance);
        }
        return (covariance - linear_operator->covariance()).lpNorm<Eigen::Infinity>();
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
    std::shared_ptr<CholeskySampler> sampler = std::make_shared<CholeskySampler>(*linear_operator, rng);
    double error = covariance_error(linear_operator, sampler);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error, 0.0, tolerance);
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
    std::shared_ptr<CholeskySampler> sampler = std::make_shared<CholeskySampler>(*linear_operator, rng);
    double error = covariance_error(linear_operator, sampler);
    const double tolerance = 2.E-3;
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_SAMPLER_HH