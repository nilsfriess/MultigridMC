#ifndef TEST_CHOLMOD_WRAPPER_HH
#define TEST_CHOLMOD_WRAPPER_HH TEST_CHOLMOD_WRAPPER_HH

#include <utility>
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cholmod_wrapper.hh"
#include "cholmod.h"

/** @brief fixture class for Cholmod tests */
class CholmodWrapperTest : public ::testing::Test
{
protected:
    /** @brief initialise tests */
    void SetUp() override
    {
        const unsigned int nrow = 16;
        A_sparse = Eigen::SparseMatrix<double>(nrow, nrow);
        typedef Eigen::Triplet<double> T;
        std::vector<T> triplet_list;

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
    }

protected:
    /** @brief Sparse matrix that will be Cholesky factorised */
    Eigen::SparseMatrix<double> A_sparse;
};

/* Test Cholmod solver using a monolithic solve */
TEST_F(CholmodWrapperTest, TestSolve)
{
    const double tolerance = 1.E-12;
    unsigned int nrow = A_sparse.rows();
    unsigned int ncol = A_sparse.cols();
    int seed = 1212957;
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal_dist;
    Eigen::VectorXd x(nrow);
    Eigen::VectorXd x_exact(nrow);
    for (unsigned int j = 0; j < nrow; ++j)
    {
        x_exact(j) = normal_dist(rng);
    }
    Eigen::VectorXd b = A_sparse * x_exact;
    x.setZero();
    CholmodLLT llt(A_sparse);
    llt.solve(b, x);
    double error = (x - x_exact).norm();

    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test Cholmod solver using two subsequent solves */
TEST_F(CholmodWrapperTest, TestSolveLLT)
{
    const double tolerance = 1.E-12;
    unsigned int nrow = A_sparse.rows();
    unsigned int ncol = A_sparse.cols();
    int seed = 1212957;
    std::mt19937 rng(seed);
    std::normal_distribution<double> normal_dist;

    Eigen::VectorXd x(nrow);
    Eigen::VectorXd y(nrow);
    Eigen::VectorXd x_exact(nrow);
    for (unsigned int j = 0; j < nrow; ++j)
    {
        x_exact(j) = normal_dist(rng);
    }
    Eigen::VectorXd b = A_sparse * x_exact;
    x.setZero();
    CholmodLLT llt(A_sparse);
    llt.solveL(b, y);
    llt.solveLT(y, x);
    double error = (x - x_exact).norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_CHOLMOD_WRAPPER_HH