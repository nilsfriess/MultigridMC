#include "cholesky_wrapper.hh"

/** @file cholmod_wrapper.cc
 *
 * @brief Implementation of cholmod_wrapper.hh
 */

/* Constructor */
CholmodLLT::CholmodLLT(const MatrixType &matrix) : nrow(matrix.rows()),
                                                   ncol(matrix.cols())
{
    const MatrixType A_lower = MatrixType(matrix.triangularView<Eigen::Lower>());
    unsigned int nnz = A_lower.nonZeros();
    ctx = new cholmod_common;
    cholmod_start(ctx);
    A_cholmod = cholmod_allocate_sparse(
        nrow, ncol, nnz,
        false,
        true,
        -1,
        CHOLMOD_REAL,
        ctx);

    std::copy_n(A_lower.outerIndexPtr(), nrow + 1, (int *)A_cholmod->p);
    std::copy_n(A_lower.innerIndexPtr(), nnz, (int *)A_cholmod->i);
    std::copy_n(A_lower.valuePtr(), nnz, (double *)A_cholmod->x);
    ctx->final_ll = true;
    L_cholmod = cholmod_analyze(A_cholmod, ctx);
    cholmod_factorize(A_cholmod, L_cholmod, ctx);
}

/* Solve full system LL^T x = b */
void CholmodLLT::solve(const Eigen::VectorXd &b, Eigen::VectorXd &x) const
{
    cholmod_dense *b_cholmod = cholmod_allocate_dense(nrow, 1, nrow, CHOLMOD_REAL, ctx);
    std::copy_n(&b[0], nrow, (double *)b_cholmod->x);
    cholmod_dense *Pb_cholmod = cholmod_solve(CHOLMOD_P, L_cholmod, b_cholmod, ctx);
    cholmod_dense *y_cholmod = cholmod_solve(CHOLMOD_L, L_cholmod, Pb_cholmod, ctx);
    cholmod_dense *Px_cholmod = cholmod_solve(CHOLMOD_Lt, L_cholmod, y_cholmod, ctx);
    cholmod_dense *x_cholmod = cholmod_solve(CHOLMOD_Pt, L_cholmod, Px_cholmod, ctx);
    std::copy_n((double *)x_cholmod->x, nrow, &x[0]);
    cholmod_free_dense(&Pb_cholmod, ctx);
    cholmod_free_dense(&y_cholmod, ctx);
    cholmod_free_dense(&Px_cholmod, ctx);
    cholmod_free_dense(&x_cholmod, ctx);
}

/* Solve lower triangular system L x = b */
void CholmodLLT::solveL(const Eigen::VectorXd &b, Eigen::VectorXd &x) const
{
    cholmod_dense *b_cholmod = cholmod_allocate_dense(nrow, 1, nrow, CHOLMOD_REAL, ctx);
    std::copy_n(&b[0], nrow, (double *)b_cholmod->x);
    cholmod_dense *Pb_cholmod = cholmod_solve(CHOLMOD_P, L_cholmod, b_cholmod, ctx);
    cholmod_dense *x_cholmod = cholmod_solve(CHOLMOD_L, L_cholmod, Pb_cholmod, ctx);
    std::copy_n((double *)x_cholmod->x, nrow, &x[0]);
    cholmod_free_dense(&Pb_cholmod, ctx);
    cholmod_free_dense(&x_cholmod, ctx);
    cholmod_free_dense(&b_cholmod, ctx);
}

/* Solve upper triangular system L^T x = b */
void CholmodLLT::solveLT(const Eigen::VectorXd &b, Eigen::VectorXd &x) const
{
    cholmod_dense *b_cholmod = cholmod_allocate_dense(nrow, 1, nrow, CHOLMOD_REAL, ctx);
    std::copy_n(&b[0], nrow, (double *)b_cholmod->x);
    cholmod_dense *Px_cholmod = cholmod_solve(CHOLMOD_Lt, L_cholmod, b_cholmod, ctx);
    cholmod_dense *x_cholmod = cholmod_solve(CHOLMOD_Pt, L_cholmod, Px_cholmod, ctx);
    std::copy_n((double *)x_cholmod->x, nrow, &x[0]);
    cholmod_free_dense(&Px_cholmod, ctx);
    cholmod_free_dense(&x_cholmod, ctx);
}