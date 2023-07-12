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
template <typename MatrixType>
class CholeskyLLT
{
public:
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
class CholmodLLT : public CholeskyLLT<LinearOperator::SparseMatrixType>
{
public:
    typedef CholeskyLLT<LinearOperator::SparseMatrixType> Base;
    /** @brief Constructor
     *
     * @param[in] matrix_ Matrix A to be Cholesky-factorised
     * @param[in] verbose_ Print out additional information (ignored for this class)
     */
    CholmodLLT(const LinearOperator::SparseMatrixType &matrix_,
               const bool verbose_ = false);

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
class EigenSimplicialLLT : public CholeskyLLT<LinearOperator::SparseMatrixType>
{
public:
    typedef CholeskyLLT<LinearOperator::SparseMatrixType> Base;
    typedef Eigen::SimplicialLLT<LinearOperator::SparseMatrixType,
                                 Eigen::Lower,
                                 Eigen::AMDOrdering<int>>
        SimplicialLLTType;
    /** @brief Constructor
     *
     * @param[in] matrix_ Matrix A to be Cholesky-factorised
     * @param[in] verbose_ Print out information on sparsity structure
     */
    EigenSimplicialLLT(const LinearOperator::SparseMatrixType &matrix_,
                       const bool verbose_ = false);

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

/** @brief Wrapper around Eigen's dense Cholesky factorisation
 *
 */
class EigenDenseLLT : public CholeskyLLT<LinearOperator::DenseMatrixType>
{
public:
    typedef CholeskyLLT<LinearOperator::DenseMatrixType> Base;
    typedef Eigen::LLT<LinearOperator::DenseMatrixType,
                       Eigen::Lower>
        DenseLLTType;
    /** @brief Constructor
     *
     * @param[in] matrix_ Matrix A to be Cholesky-factorised
     */
    EigenDenseLLT(const LinearOperator::DenseMatrixType &matrix_);

    /** @brief Destructor*/
    ~EigenDenseLLT() {}

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
    std::shared_ptr<DenseLLTType> dense_LLT;
};

#endif // CHOLESKY_WRAPPER_HH