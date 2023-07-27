#ifndef COMMON_HH
#define COMMON_HH COMMON_HH

#include <vector>

/** @file common.hh
 *
 * @brief contains commonly used, generic auxilliary routines
 */

/** @brief Construct n-fold Cartesian product of vector with itself
 *
 * Given a vector v = (v_0,v_1,...,v_{d-1}) with d elements, this method
 * constructs the d^n dimensional vector w = (w_1,w_2,...,w_{d^n-1}) such
 * that the entries w_j = (v_{j_0},v_{j_1},..v_{j_{d-1}}) contain all
 * possible combinations of the elements of v.
 *
 * For example for v = (0,1) and n = 3 the result would be the 8-dimensional
 * vector
 *
 *    w = ((0,0,0),(0,0,1),(0,1,0),(0,1,1),(1,0,0),(1,0,1),(1,1,0),(1,1,1))
 *
 * If n=0, an empty vector w = (()) will be returned.
 *
 * @param[in] v vector v
 * @param[in] n order of the cartesian product
 */
template <class T>
std::vector<std::vector<T>> cartesian_product(std::vector<T> v, int n)
{
    std::vector<std::vector<T>> prod;
    if (n == 1)
    {
        for (auto v_it = v.begin(); v_it != v.end(); ++v_it)
        {
            prod.push_back(std::vector<T>{*v_it});
        }
    }
    else
    {
        std::vector<std::vector<T>> prod_prev = cartesian_product(v, n - 1);
        for (auto s_it = prod_prev.begin(); s_it != prod_prev.end(); ++s_it)
        {
            for (auto v_it = v.begin(); v_it != v.end(); ++v_it)
            {
                std::vector<T> s_j = *s_it;
                s_j.push_back(*v_it);
                prod.push_back(s_j);
            }
        }
    }
    return prod;
}
#endif // COMMON_HH