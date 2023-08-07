#ifndef SQUARED_SHIFTEDLAPLACE_FD_OPERATOR_HH
#define SQUARED_SHIFTEDLAPLACE_FD_OPERATOR_HH SQUARED_SHIFTEDLAPLACE_FD_OPERATOR_HH

#include <utility>
#include <vector>
#include <Eigen/Dense>
#include "auxilliary/common.hh"
#include "linear_operator.hh"
#include "correlationlength_model.hh"
#include "lattice/lattice.hh"

/** @file squared_shiftedlaplace_fd_operator.hh
 *
 * @brief Finite difference discretisation of the squared shifted laplace operator in arbitrary dimensions
 */

/** @class SquaredShiftedLaplaceFDOperator
 *
 * Class for finite difference discretisation of shifted biharmonic operator
 *
 *     (-Laplace + kappa^{-2})^2 u
 *   = Laplace^2(u) -2 * kappa^{-2} Laplace(u) + kappa^{-4} u
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
 *
 * inside the domain. At the boundary of the domain, the stencil needs to be modified to
 *
 *                        C'
 *                        !
 *                        !
 *      c                 B' --- D
 *      !                 !      !
 *      !                 !      !
 *      a ---- b         A+C --- B ---- C
 *      !                 !      !
 *      !                 !      !
 *      c                 B' --- D
 *                        !
 *                        !
 *                        C'
 *
 * Note that at the boundary the discretisation has an O(1/h) error [2], i.e.
 *
 *   D_4 u = Laplace^2 u + O(1/h).
 *
 * This error can be reduced to O(h) by replacing A+C -> A + 6C, B -> B-2C, C->C+1/3 C, but the
 * resulting operator is non-symmetric, which is why this approach is not chosen here. The idea for
 * deriving this expression is to use a quartic extrapolation x^2*(a+b*x+c*x^2) to extrapolate the
 * ghost point outside the domain. In [2] an alternative approach is described, which uses a cubic
 * extrapolation to obtain an O(1) error is described.
 *
 * The biharmonic equation discretised with finite differences is solved with multigrid methods in [3-6].
 *
 *   [1] Bjorstad, P., 1983. Fast numerical solution of the biharmonic Dirichlet problem on rectangles.
 *       SIAM Journal on Numerical Analysis, 20(1), pp.59-71.
 *   [2] Bjorstad, P.E., 1981. Numerical solution of the biharmonic equation. Stanford University.
 *       (PhD thesis)
 *   [3] Dehghan, M. and Mohebbi, A., 2008. Solution of the two dimensional second biharmonic equation
 *       with high‐order accuracy. Kybernetes, 37(8), pp.1165-1179.
 *   [4] Linden, J., 1985. A multigrid method for solving the biharmonic equation on rectangular domains.
 *       In Advances in Multi-Grid Methods: Proceedings of the conference held in Oberwolfach, December 8 to 13,
 *       1984 (pp. 64-76). Vieweg+ Teubner Verlag.
 *   [5] Pan, K., He, D. and Ni, R., 2019. An efficient multigrid solver for 3D biharmonic equation with
 *       a discretization by 25-point difference scheme. arXiv preprint arXiv:1901.05118.
 *   [6] Ibrahim, S.A. and Hassan, N.A., 2012. MULTIGRID SOLUTION OF THREE DIMENSIONAL BIHARMONIC EQUATIONS
 *       WITH DIRICHLET BOUNDARY CONDITIONS OF SECOND KIND. Journal of applied mathematics & informatics,
 *       30(1_2), pp.235-244.
 *
 */
class SquaredShiftedLaplaceFDOperator : public LinearOperator
{
public:
    /** @brief Create a new instance
     *
     * Populates matrix entries across the grid
     *
     * @param[in] lattice_ underlying 2d lattice
     * @param[in] correlationlength_model_ model for correlation length in domain
     * @param[in] verbose_ verbosity level
     */
    SquaredShiftedLaplaceFDOperator(const std::shared_ptr<Lattice> lattice_,
                                    const std::shared_ptr<CorrelationLengthModel> correlationlength_model_,
                                    const int verbose = 0);

protected:
    /** @brief Model for correlation length in domain */
    const std::shared_ptr<CorrelationLengthModel> correlationlength_model;
};

#endif // SQUARED_SHIFTEDLAPLACE_FD_OPERATOR_HH