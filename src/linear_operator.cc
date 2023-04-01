#include "linear_operator.hh"
/** @file linear_operator.cc
 *
 * @brief Implementation of linear_operator.hh
 *
 * Defines the specific offsets for operators
 */

/** @brief Offsets in x-direction for 5point operator in 2d */
template <>
const int BaseLinearOperator2d<5, LinearOperator2d5pt>::offset_x[5] = {0, 0, 0, -1, +1};
/** @brief Offsets in y-direction for 5point operator in 2d */
template <>
const int BaseLinearOperator2d<5, LinearOperator2d5pt>::offset_y[5] = {0, -1, +1, 0, 0};
