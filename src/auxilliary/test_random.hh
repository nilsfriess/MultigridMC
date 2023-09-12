#include <gtest/gtest.h>
#include "omp.h"
#include <cmath>
#include <vector>
#include <numeric>
#include "parallel_random.hh"

/** @brief test random number generator class
 *

 */
class RandomTest : public ::testing::Test
{
public:
    /** @Create a new instance */
    RandomTest() {}

protected:
    /** @brief initialise tests */
    void
    SetUp() override
    {
    }
};

/** @brief check that RNG generates the same random numbers in parallel as in serial */
TEST_F(RandomTest, TestParallelConsistent)
{
    const int n_samples = 64;
    CombinedLinearCongruentialGenerator rng_sequential;
    std::vector<int> a_sequential(n_samples);
    std::vector<int> a_parallel(n_samples);
    for (int j = 0; j < n_samples; ++j)
        a_sequential[j] = rng_sequential.draw_int();
#pragma omp parallel num_threads(4)
    {
        int n_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
#pragma omp master
        {
            if (n_threads == 1)
            {
                std::cout << "Warning: running on one thread only!" << std::endl;
            }
        }
        CombinedLinearCongruentialGenerator rng_parallel;
        for (int j = thread_id; j < n_samples; j += n_threads)
            a_parallel[j] = rng_parallel.draw_int();
    }
    EXPECT_TRUE(a_sequential == a_parallel);
}

/** @brief check that RNG generates uniform random numbers correctly */
TEST_F(RandomTest, TestUniform)
{
    const int n_samples = 1 << 24;
    const int n_bins = 32;
    CombinedLinearCongruentialGenerator rng;
    std::vector<double> histogram(n_bins);
    double x_min = 1.1;
    double x_max = 3.4;
    for (int j = 0; j < n_samples; ++j)
    {
        double x = x_min + rng.draw_uniform_real() * (x_max - x_min);
        int idx = std::max(int(floor((x - x_min) / (x_max - x_min) * n_bins)), 0);
        histogram[idx] += 1.0;
    }
    for (int k = 0; k < n_bins; ++k)
        histogram[k] *= 1 / double(n_samples);
    std::vector<double> cdf(n_samples);
    std::partial_sum(histogram.begin(), histogram.end(), cdf.begin());
    double max_diff = 0.0;
    for (int k = 0; k < n_bins; ++k)
    {
        double cdf_true = (k + 1) / double(n_bins);
        max_diff = std::max(max_diff, fabs(cdf_true - cdf[k]));
    }
    double tolerance = 2.E-4;
    EXPECT_NEAR(max_diff, 0.0, tolerance);
}

/** @brief check that RNG generates normal random numbers correctly */
TEST_F(RandomTest, TestNormal)
{
    const int n_samples = 1 << 24;
    const int n_bins = 64;
    CombinedLinearCongruentialGenerator rng;
    std::vector<double> histogram(n_bins);
    double x_min = -10.0;
    double x_max = +10.0;
    for (int j = 0; j < n_samples; ++j)
    {
        double x = rng.draw_normal();
        int idx = std::max(int(floor((x - x_min) / (x_max - x_min) * n_bins)), 0);
        histogram[idx] += 1.0;
    }
    for (int k = 0; k < n_bins; ++k)
        histogram[k] *= 1 / double(n_samples);
    std::vector<double> cdf(n_samples);
    std::partial_sum(histogram.begin(), histogram.end(), cdf.begin());
    double max_diff = 0.0;
    for (int k = 0; k < n_bins; ++k)
    {
        double x = x_min + (k + 1) / double(n_bins) * (x_max - x_min);
        double cdf_true = 0.5 * (1.0 + erf(x / sqrt(2.0)));
        max_diff = std::max(max_diff, fabs(cdf_true - cdf[k]));
    }
    double tolerance = 2.E-4;
    EXPECT_NEAR(max_diff, 0.0, tolerance);
}