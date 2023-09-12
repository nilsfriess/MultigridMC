#ifndef PARALLEL_RANDOM_HH
#define PARALLEL_RANDOM_HH PARALLEL_RANDOM_HH
/** @file parallel_random.hh
 *
 * @brief thread-safe random number generator
 */

#include <stdint.h>
#include <limits>
#include <cmath>
#include <random>
#include <utility>
#include "omp.h"

/** @class CombinedLinearCongruentialGenerator
 *
 * @brief Thread-safe Combined Linear congruential generator
 *
 * The update rule of the two underlying linear congruential generators is:
 *
 *  y_{1,n+1} = ( a_1 * y_{1,n} ) mod m_1
 *  y_{2,n+1} = ( a_2 * y_{2,n} ) mod m_2
 *
 * for some seeds y_{1,0}, y_{2,0}.
 *
 * In the parallel generator, thread k will use the random numbers
 *
 *   x_{k}, x_{k+n_thread},  x_{k+2*n_thread}, ...
 *
 * and this guarantees that each thread uses a non-overlapping sequence.
 *
 * For the combined linear congruential generator, see
 *      L'Ecuyer, P., 1988: "Efficient and portable combined random number generators."
 *      Communications of the ACM, 31(6), pp.742-751.
 *      http://www.iro.umontreal.ca/~lecuyer/myftp/papers/cacm88.pdf
 *
 */
class CombinedLinearCongruentialGenerator
{
public:
    /** @brief Constructor
     *
     * @param[in] a1 multiplier a in update rule of first LCG
     * @param[in] m1 period in update rule of first LCG
     * @param[in] seed1 seed of first LCG
     * @param[in] a2 multiplier a in update rule of second LCG
     * @param[in] m2 period in update rule of second LCG
     * @param[in] seed2 seed of second LCG
     */
    CombinedLinearCongruentialGenerator();

    /** @brief draw integer random number */
    int64_t draw_int();

    /** @brief draw uniform real random number
     *
     * Return a uniformly distributed number from the interval ]0,1[
     * (not including the end points)
     */
    double draw_uniform_real();

    /** @brief draw normal real random number
     *
     * Return a random number from the normal distribution N(0,1) with density
     *
     *    rho(x) = 1/sqrt(2*pi)*exp(-x^2/2)
     */
    double draw_normal();

    /** @brief state y_{1,n} of first LCG */
    int64_t y1;
    /** @brief state y_{2,n} of second LCG */
    int64_t y2;
    /** @brief multiplier a_1 of first LCG */
    const int64_t a1;
    /** @brief period m_1 of first LCG */
    const int64_t m1;
    /** @brief inverse 1/m_1 of m_1*/
    const double m1_inv;
    /** @brief seed y_{1,0} of first LCG */
    const int64_t seed1;
    /** @brief multiplier a_1 of second LCG */
    const int64_t a2;
    /** @brief period m_2 of second LCG */
    const int64_t m2;
    /** @brief seed y_{2,0} of second LCG */
    const int64_t seed2;
    /** @brief multiplier a_1^{n_threads} mod m_1 for first LCG in parallel case */
    int64_t a1_multistep;
    /** @brief multiplier a_2^{n_threads} mod m_2 for second LCG in parallel case */
    int64_t a2_multistep;
    /** @brief largest random number when drawing from interval ]0,1[; x_max = (m_1-1)/m_1*/
    const double x_max;
    /** @brief machine epsilon */
    const double epsilon;
    /** @brief 2*pi */
    const double two_pi;
};

#endif // PARALLEL_RANDOM_HH