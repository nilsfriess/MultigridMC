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
    LinearOperator(const std::shared_ptr<Lattice> lattice_) : lattice(lattice_),
                                                              A_sparse(lattice_->M, lattice_->M)
    {
    }

    /** @brief Create a new instance from a matrix
     *
     * @param[in] lattice_ underlying lattice
     */
    LinearOperator(const std::shared_ptr<Lattice> lattice_,
                   const SparseMatrixType &A_sparse_) : lattice(lattice_),
                                                        A_sparse(A_sparse_) {}

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

    /** @brief Convert to sparse storage format */
    const SparseMatrixType &to_sparse() const { return A_sparse; };

protected:
    /** @brief underlying lattice */
    const std::shared_ptr<Lattice> lattice;
    SparseMatrixType A_sparse;
};

#endif // LINEAR_OPERATOR_HH
