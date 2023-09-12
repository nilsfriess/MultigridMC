#include "parallel_random.hh"

/** @file parallel_random.cc
 *
 * @brief Implementation of parallel_random.hh
 * */

/* Constructor */
CombinedLinearCongruentialGenerator::CombinedLinearCongruentialGenerator() : a1(40014LL),
                                                                             m1(2147483563LL),
                                                                             m1_inv(1. / 2147483563.),
                                                                             seed1(1LL),
                                                                             a2(40692LL),
                                                                             m2(2147483399LL),
                                                                             seed2(1LL),
                                                                             x_max(double(m1 - 1) / double(m1)),
                                                                             epsilon(std::numeric_limits<double>::epsilon()),
                                                                             two_pi(2.0 * M_PI)
{
    int n_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    a1_multistep = a1;
    a2_multistep = a2;
    y1 = seed1;
    y2 = seed2;
    for (int j = 1; j < n_threads; ++j)
    {
        if (j <= thread_id)
        {
            y1 = (a1 * y1) % m1;
            y2 = (a2 * y2) % m2;
        }
        a1_multistep = (a1_multistep * a1) % m1;
        a2_multistep = (a2_multistep * a2) % m2;
    }
}

/* draw integer random number */
int64_t CombinedLinearCongruentialGenerator::draw_int()
{
    int64_t x = (y1 - y2 + m1 - 1) % (m1 - 1);
    y1 = (a1_multistep * y1) % m1;
    y2 = (a2_multistep * y2) % m2;
    return x;
}

/* draw uniform real random number */
double CombinedLinearCongruentialGenerator::draw_uniform_real()
{
    int64_t x = draw_int();
    if (x == 0LL)
    {
        return x_max;
    }
    else
    {
        return x * m1_inv;
    }
}

/* draw Gaussian real random number */
double CombinedLinearCongruentialGenerator::draw_normal()
{
    double u1, u2;
    do
    {
        u1 = draw_uniform_real();
    } while (u1 <= epsilon);
    u2 = draw_uniform_real();
    return sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
}