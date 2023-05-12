#ifndef CHOLMOD_WRAPPER_HH
#define CHOLMOD_WRAPPER_HH CHOLMOD_WRAPPER_HH
#include <algorithm>
#include "cholmod.h"
#include "linear_operator/linear_operator.hh"

/** @file cholmod_wrapper.hh
 *
 * @brief Wrapper around Cholmod's Supernodal Cholesky factorisation
 *
 */

/** @brief Wrapper around Cholmod's Cholesky factorisation
 *
 * Given the Cholesky factorisation A=LL^T of a matrix, this class provides methods for
 * solving Lx = b and L^Tx = b.
 */
class CholmodLLT
{
public:
    typedef LinearOperator::SparseMatrixType MatrixType;
    /** @brief Constructor
     *
     * @param[in] matrix Matrix A to be Cholesky-factorised
     */
    CholmodLLT(const MatrixType &matrix);

    /** @brief Destructor*/
    ~CholmodLLT()
    {
        cholmod_free_sparse(&A_cholmod, ctx);
        cholmod_free_factor(&L_cholmod, ctx);
        cholmod_finish(ctx);
        delete ctx;
    }

    /** @brief Solve the full linear system LL^T x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    void solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    /** @brief Solve the lower triangular system L x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    void solveL(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    /** @brief Solve the upper triangular system L^T x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    void solveLT(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

protected:
    /** @brief number of matrix rows */
    const unsigned int nrow;
    /** @brief number of matrix columns */
    const unsigned int ncol;
    /** @brief Cholmod common context */
    cholmod_common *ctx;
    /** @brief cholmod matrix */
    cholmod_sparse *A_cholmod;
    /** @brief Cholmod factor */
    cholmod_factor *L_cholmod;
};
#endif // CHOLMOD_WRAPPER_HH