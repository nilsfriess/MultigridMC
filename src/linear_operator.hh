#ifndef LINEAR_OPERATOR_HH
#define LINEAR_OPERATOR_HH LINEAR_OPERATOR_HH
#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "lattice.hh"
#include "samplestate.hh"

/** @file LinearOperator.hh
 * @brief Header file for LinearOperator classes
 */

/** @class LinearOperator
 *
 * @brief LinearOperator class
 *
 * A linear operator class provides the following functionality:
 *
 *  - application of the operator, i.e. the linear map L : x -> y
 *
 *       y_j = sum_{k=1}^{N} A_{jk} x_k
 *
 *  - Gibbs-sampling with the corresponding matrix A, i.e. for all j=1,...,N
 *
 *       x_j -> (b_j - sum_{k != j} A_{jk} x_k) / A_{jj} + \xi_j / \sqrt(A_{jj})
 *
 *    where \xi ~ N(0,1) is a normal variable
 *
 * The sparsity structure of the matrix is assumed to be fixed and is described
 * by a stencil.
 *
 */
class LinearOperator
{
public:
    /** @brief Type of sparse matrix*/
    typedef Eigen::SparseMatrix<double> SparseMatrixType;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice
     * @param[in] rng random number generator
     */
    LinearOperator(const std::shared_ptr<Lattice> lattice_,
                   std::mt19937_64 &rng_) : lattice(lattice_),
                                            rng(rng_),
                                            normal_dist(0.0, 1.0),
                                            A_sparse(lattice_->M, lattice_->M)
    {
        sqrt_inv_diag = new double[lattice->M];
    }

    /** @brief Create a new instance from a matrix
     *
     * @param[in] lattice_ underlying lattice
     * @param[in] rng random number generator
     */
    LinearOperator(const std::shared_ptr<Lattice> lattice_,
                   std::mt19937_64 &rng_,
                   const SparseMatrixType &A_sparse_) : lattice(lattice_),
                                                        rng(rng_),
                                                        normal_dist(0.0, 1.0),
                                                        A_sparse(A_sparse_)
    {
        set_inv_sqrt_diagonal();
    }

    /** @brief Compute 1/sqrt(A_diag) */
    void set_inv_sqrt_diagonal()
    {
        auto diag = A_sparse.diagonal();
        for (unsigned ell = 0; ell < lattice->M; ++ell)
        {
            sqrt_inv_diag[ell] = 1 / sqrt(diag[ell]);
        }
    }

    /** @brief Destroy instance */
    ~LinearOperator()
    {
        delete[] sqrt_inv_diag;
    }

    /** @brief Extract underlying lattice */
    std::shared_ptr<Lattice> get_lattice() const
    {
        return lattice;
    }

    /** @brief Apply the linear LinearOperator
     *
     * Compute y = Ax
     *
     * @param[in] x input state
     * @param[out] y output state
     */
    void apply(const std::shared_ptr<SampleState> x, std::shared_ptr<SampleState> y)
    {
        y->data = A_sparse * x->data;
    }

    /** @brief Carry out a single Gibbs-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    void gibbssweep(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x)
    {
        const auto row_ptr = A_sparse.outerIndexPtr();
        const auto col_ptr = A_sparse.innerIndexPtr();
        const auto val_ptr = A_sparse.valuePtr();
        unsigned int nrow = A_sparse.rows();
        for (int ell = 0; ell < nrow; ++ell)
        {
            double residual = 0.0;
            for (int k = row_ptr[ell]; k < row_ptr[ell + 1]; ++k)
            {
                unsigned int ell_prime = col_ptr[k];
                residual += val_ptr[k] * x->data[ell_prime];
            }
            const double a_sqrt_inv_diag = sqrt_inv_diag[ell];
            // subtract diagonal contribution
            residual -= x->data[ell] / (a_sqrt_inv_diag * a_sqrt_inv_diag);
            x->data[ell] = ((b->data[ell] - residual) * a_sqrt_inv_diag + normal_dist(rng)) * a_sqrt_inv_diag;
        }
    }

    /** @brief Convert to sparse storage format */
    const SparseMatrixType &to_sparse() const { return A_sparse; };

    /** @brief Get RNG*/
    std::mt19937_64 &get_rng() const { return rng; }

protected:
    /** @brief underlying lattice */
    const std::shared_ptr<Lattice> lattice;
    /** @brief random number generator */
    std::mt19937_64 &rng;
    /** @brief normal distribution for Gibbs-sweep */
    std::normal_distribution<double> normal_dist;
    /** @brief Underlying sparse Eigen matrix */
    SparseMatrixType A_sparse;
    /** @brief square root of inverse diagonal matrix entries (required in Gibbs sweep) */
    double *sqrt_inv_diag;
};

#endif // LINEAR_OPERATOR_HH
