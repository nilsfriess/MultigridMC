#include "parallel_random.hh"

/** @file parallel_random.cc
 *
 * @brief Implementation of parallel_random.hh
 * */

/* draw Gaussian real random number */
double RandomGenerator::draw_normal()
{
    if (box_muller_redraw)
    {
        double u1, u2;
        double s;
        do
        {
            u1 = 2. * draw_uniform_real() - 1.;
            u2 = 2. * draw_uniform_real() - 1.;
            s = u1 * u1 + u2 * u2;
        } while ((s == 0) or (s >= 1));
        double L = sqrt(-2 * log(s) / s);
        z_1 = L * u1;
        z_2 = L * u2;
        box_muller_redraw = false;
        return z_1;
    }
    else
    {
        box_muller_redraw = true;
        return z_2;
    }
}

/* Constructor */
CLCGenerator::CLCGenerator() : a1(40014),
                               m1(2147483563),
                               m1_inv(1. / 2147483563.),
                               seed1(1),
                               a2(40692),
                               m2(2147483399),
                               seed2(1),
                               x_max(double(m1 - 1) / double(m1))
{
    int n_threads = omp_get_num_threads();
    int thread_id = omp_get_thread_num();
    a1_multistep = (int64_t)a1;
    a2_multistep = (int64_t)a2;
    y1 = seed1;
    y2 = seed2;
    for (int j = 1; j < n_threads; ++j)
    {
        if (j <= thread_id)
        {
            y1 = ((int64_t)a1 * y1) % m1;
            y2 = ((int64_t)a2 * y2) % m2;
        }
        a1_multistep = (a1_multistep * a1) % m1;
        a2_multistep = (a2_multistep * a2) % m2;
    }
}

/* draw integer random number */
int32_t CLCGenerator::draw_int()
{
    int32_t x = y1 - y2;
    if (x < 1)
        x += m1 - 1;
    y1 = (a1_multistep * y1) % m1;
    y2 = (a2_multistep * y2) % m2;
    return x;
}

/* draw uniform real random number */
double CLCGenerator::draw_uniform_real()
{
    int32_t x = draw_int();
    return x * m1_inv;
}