#ifndef TEST_CHOLMOD_WRAPPER_HH
#define TEST_CHOLMOD_WRAPPER_HH TEST_CHOLMOD_WRAPPER_HH

#include <utility>
#include <algorithm>
#include <gtest/gtest.h>
#include <random>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "cholesky_wrapper.hh"

/** @brief fixture class for Cholmod tests */
class CholeskyWrapperTest : public ::testing::Test
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
        A_dense = Eigen::MatrixXd(A_sparse);
        // Construct solution, RHS and numerical solution
        int seed = 1212957;
        std::mt19937 rng(seed);
        std::normal_distribution<double> normal_dist;
        x = Eigen::VectorXd(nrow);
        x_exact = Eigen::VectorXd(nrow);
        for (unsigned int j = 0; j < nrow; ++j)
        {
            x_exact(j) = normal_dist(rng);
        }
        b = A_sparse * x_exact;
    }

protected:
    /** @brief Sparse matrix that will be Cholesky factorised */
    Eigen::SparseMatrix<double> A_sparse;
    /** @brief Dense representation of matrix that will be Cholesky factorised */
    Eigen::MatrixXd A_dense;
    /** @brief Numerical solution*/
    Eigen::VectorXd x;
    /** @brief Exact solution*/
    Eigen::VectorXd x_exact;
    /** @brief right hand side A x_{exact} = b*/
    Eigen::VectorXd b;
};

/* Test Simplicial Cholesky solver using a monolithic solve */
TEST_F(CholeskyWrapperTest, TestEigenSimplicialSolve)
{
    const double tolerance = 1.E-12;
    EigenSimplicialLLT llt(A_sparse);
    llt.solve(b, x);
    double error = (x - x_exact).norm();

    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test Simplicial Cholesky solver using two subsequent solves */
TEST_F(CholeskyWrapperTest, TestEigenSimplicialSolveLLT)
{
    const double tolerance = 1.E-12;
    Eigen::VectorXd y(x.size());
    EigenSimplicialLLT llt(A_sparse);
    llt.solveL(b, y);
    llt.solveLT(y, x);
    double error = (x - x_exact).norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test Cholmod solver using a monolithic solve */
TEST_F(CholeskyWrapperTest, TestCholmodSolve)
{
#ifndef NCHOLMOD
    const double tolerance = 1.E-12;
    CholmodLLT llt(A_sparse);
    llt.solve(b, x);
    double error = (x - x_exact).norm();

    EXPECT_NEAR(error, 0.0, tolerance);
#else
    GTEST_SKIP();
#endif // NCHOLMOD
}

/* Test Cholmod solver using two subsequent solves */
TEST_F(CholeskyWrapperTest, TestCholmodSolveLLT)
{
#ifndef NCHOLMOD
    const double tolerance = 1.E-12;
    Eigen::VectorXd y(x.size());
    CholmodLLT llt(A_sparse);
    llt.solveL(b, y);
    llt.solveLT(y, x);
    double error = (x - x_exact).norm();
    EXPECT_NEAR(error, 0.0, tolerance);
#else
    GTEST_SKIP();
#endif // NCHOLMOD
}

/* Test dense Cholesky solver using a monolithic solve */
TEST_F(CholeskyWrapperTest, TestEigenDenseSolve)
{
    const double tolerance = 1.E-12;
    EigenDenseLLT llt(A_dense);
    llt.solve(b, x);
    double error = (x - x_exact).norm();

    EXPECT_NEAR(error, 0.0, tolerance);
}

/* Test dense Cholesky solver using two subsequent solves */
TEST_F(CholeskyWrapperTest, TestEigenDenseSolveLLT)
{
    const double tolerance = 1.E-12;
    Eigen::VectorXd y(x.size());
    EigenDenseLLT llt(A_dense);
    llt.solveL(b, y);
    llt.solveLT(y, x);
    double error = (x - x_exact).norm();
    EXPECT_NEAR(error, 0.0, tolerance);
}

#endif // TEST_CHOLMOD_WRAPPER_HH