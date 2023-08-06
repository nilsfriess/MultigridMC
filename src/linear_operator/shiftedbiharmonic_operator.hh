#ifndef SHIFTEDBIHARMONIC_OPERATOR_HH
#define SHIFTEDBIHARMONIC_OPERATOR_HH SHIFTEDBIHARMONIC_OPERATOR_HH

#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "auxilliary/common.hh"
#include "linear_operator.hh"
#include "lattice/lattice.hh"

/** @file shiftedlaplace_operator.hh
 *
 * @brief Contains class for shifted laplace operator in arbitrary dimensions
 */

/** @class ShiftedLaplaceOperator
 *
 * Class for finite difference discretisation of shifted biharmonic operator
 *
 *     (-alpha_K * Laplace + alpha_b)^2 u
 *   = alpha_K^2 Laplace^2(u) -2 * alpha_K * alpha_b Laplace(u) + alpha_b^2 u
 *
 *
 * with homogeneous Dirichlet boundary conditions u(x)=0, du/dn(x)=0 on the boundary.
 *
 * The finite difference stencils for the Laplace operator and of the squared Laplace operator are
 * given in two dimensions by
 *
 *
 *                                                 C'
 *                                                 !
 *                                                 !
 *             c                            D ---- B' --- D
 *             !                            !      !      !
 *             !                            !      !      !
 *      b ---- a ---- b              C ---- B ---- A ---- B ---- C
 *             !                            !      !      !
 *             !                            !      !      !
 *             c                            D ---- B' --- D
 *                                                 !
 *                                                 !
 *                                                 C'
 *
 * With
 *
 *    a = -2 * (1/h_x^2 + 1/h_y^2)
 *    b = 1/h_x^2
 *    c = 1/h_y^2
 *
 *    A  = 6 * (1/h_x^4 + 1/h_y^4) + 8/(h_x^2*h_y^2)
 *    B  = -4/h_x^2 * (1/h_x^2 + 1/h_y^2)
 *    B' = -4/h_y^2 * (1/h_x^2 + 1/h_y^2)
 *    C  = 1/h_x^4
 *    C' = 1/h_y^4
 *    D  = 2/(h_x^2*h_y^2)
 */
class ShiftedBiharmonicOperator : public LinearOperator
{
public:
    /** @brief Create a new instance
     *
     * Populates matrix entries across the grid
     *
     * @param[in] lattice_ underlying 2d lattice
     * @param[in] alpha_K coefficient of second order term
     * @param[in] alpha_b coefficient of zero order term
     * @param[in] verbose_ verbosity level
     */
    ShiftedBiharmonicOperator(const std::shared_ptr<Lattice> lattice_,
                              const double alpha_K_,
                              const double alpha_b_,
                              const int verbose = 0);

protected:
    /** @brief Coefficient of Laplace term */
    const double alpha_K;
    /** @brief Coefficient of zero order term */
    const double alpha_b;
};

#endif // SHIFTEDBIHARMONIC_OPERATOR_HH