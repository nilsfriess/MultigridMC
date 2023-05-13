#ifndef CHOLESKY_WRAPPER_HH
#define CHOLESKY_WRAPPER_HH CHOLESKY_WRAPPER_HH
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>
#include "config.h"
#ifndef NCHOLMOD
#include "cholmod.h"
#endif // NCHOLMOD
#include "linear_operator/linear_operator.hh"

/** @file cholesky_wrapper.hh
 *
 * @brief Wrapper around Cholesky factorisations
 *
 */

/** @brief Abstract base class: wrapper around a Cholesky factorisation
 *
 * Given the Cholesky factorisation A=LL^T of a matrix, this class provides methods for
 * solving Lx = b and L^Tx = b as well as LL^Tx = b.
 */
class CholeskyLLT
{
public:
    typedef LinearOperator::SparseMatrixType MatrixType;

    /** @brief Constructor
     *
     * @param[in] matrix_ Matrix A to be Cholesky-factorised
     */
    CholeskyLLT(const MatrixType &matrix_) : matrix(matrix_),
                                             nrow(matrix.rows()),
                                             ncol(matrix.cols()) {}

    /** @brief Solve the full linear system LL^T x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    virtual void solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const = 0;

    /** @brief Solve the lower triangular system L x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    virtual void solveL(const Eigen::VectorXd &b, Eigen::VectorXd &x) const = 0;

    /** @brief Solve the upper triangular system L^T x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    virtual void solveLT(const Eigen::VectorXd &b, Eigen::VectorXd &x) const = 0;

protected:
    /** @brief Underlying matrix */
    const MatrixType &matrix;
    /** @brief number of matrix rows */
    const unsigned int nrow;
    /** @brief number of matrix columns */
    const unsigned int ncol;
};

/** @brief Wrapper around Cholmod's Cholesky factorisation */
#ifndef NCHOLMOD
class CholmodLLT : public CholeskyLLT
{
public:
    typedef CholeskyLLT Base;
    using Base::MatrixType;
    /** @brief Constructor
     *
     * @param[in] matrix_ Matrix A to be Cholesky-factorised
     */
    CholmodLLT(const MatrixType &matrix_);

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
    virtual void solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    /** @brief Solve the lower triangular system L x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    virtual void solveL(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    /** @brief Solve the upper triangular system L^T x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    virtual void solveLT(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    /** @brief is this using a supernodal decomposition? */
    bool is_supernodal() const { return L_cholmod->is_super; }

    /** @brief return size of matrix */
    size_t get_n() const { return L_cholmod->n; }

protected:
    /** @brief Cholmod common context */
    cholmod_common *ctx;
    /** @brief cholmod matrix */
    cholmod_sparse *A_cholmod;
    /** @brief Cholmod factor */
    cholmod_factor *L_cholmod;
};
#endif // NCHOLMOD

/** @brief Wrapper around Eigen's simplical Cholesky factorisation
 *
 */
class EigenSimplicialLLT : public CholeskyLLT
{
public:
    typedef CholeskyLLT Base;
    using Base::MatrixType;
    typedef Eigen::SimplicialLLT<LinearOperator::SparseMatrixType,
                                 Eigen::Lower,
                                 Eigen::AMDOrdering<int>>
        SimplicialLLTType;
    typedef typename Eigen::SimplicialCholeskyBase<SimplicialLLTType> EigenSimplicialCholeskyBase;
    /** @brief Constructor
     *
     * @param[in] matrix_ Matrix A to be Cholesky-factorised
     */
    EigenSimplicialLLT(const MatrixType &matrix_);

    /** @brief Destructor*/
    ~EigenSimplicialLLT() {}

    /** @brief Solve the full linear system LL^T x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    virtual void solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    /** @brief Solve the lower triangular system L x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    virtual void solveL(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

    /** @brief Solve the upper triangular system L^T x = b for x
     *
     * @param[in] b right hand side b
     * @param[out] x result x
     */
    virtual void solveLT(const Eigen::VectorXd &b, Eigen::VectorXd &x) const;

protected:
    /** @brief Cholesky factorisation */
    std::shared_ptr<SimplicialLLTType> simplicial_LLT;
};

#endif // CHOLESKY_WRAPPER_HH