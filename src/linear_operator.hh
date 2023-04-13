#ifndef LINEAR_OPERATOR_HH
#define LINEAR_OPERATOR_HH LINEAR_OPERATOR_HH
#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "lattice.hh"
#include "intergrid_operator.hh"

/** @file LinearOperator.hh
 * @brief Header file for LinearOperator classes
 */

/** @class LinearOperator
 *
 * @brief LinearOperator class
 *
 * Represents linear operators of the form
 *
 *  A = A_0 + B \Sigma^{-1} B^T
 *
 *
 * where A, A_0 are n x n matrix, \Sigma is a m x m matrix (with m << n) and
 * B is a n x m matrix.
 */

class LinearOperator
{
public:
    /** @brief Type of sparse matrix*/
    typedef Eigen::SparseMatrix<double> SparseMatrixType;
    typedef Eigen::MatrixXd DenseMatrixType;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice
     * @param[in] m_lowrank_ the dimension of the low-rank space spanned by B
     */
    LinearOperator(const std::shared_ptr<Lattice> lattice_,
                   const unsigned int m_lowrank_ = 0) : lattice(lattice_),
                                                        m_lowrank(m_lowrank_),
                                                        A_sparse(lattice_->M, lattice_->M),
                                                        B(lattice_->M, m_lowrank_),
                                                        Sigma_inv(m_lowrank_, m_lowrank_)
    {
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
    void apply(const Eigen::VectorXd &x, Eigen::VectorXd &y)
    {
        // Apply the sparse matrix A_0
        y = A_sparse * x;
        // Add low-rank correction, if necessary
        if (m_lowrank > 0)
        {
            y += B * Sigma_inv * B.transpose() * x;
        }
    }

    /** @brief Coarsen linear operator to the next-coarser level
     *
     * Compute A^{c} = I_{2h}^{h} A I_{h}^{2h}
     *
     * @param[in] intergrid_operator The intergrid operator to be used
     */
    LinearOperator coarsen(const std::shared_ptr<IntergridOperator> intergrid_operator) const;

    /** @brief Convert to sparse storage format */
    const SparseMatrixType &get_sparse() const { return A_sparse; };

    /** @brief Convert B to dense storage format */
    const DenseMatrixType &get_B() const { return B; };

    /** @brief Convert Sigma^{-1} to dense storage format */
    const DenseMatrixType &get_Sigma_inv() const { return Sigma_inv; };

protected:
    /** @brief underlying lattice */
    const std::shared_ptr<Lattice> lattice;
    /** @brief dimension of low rank space spanned by B*/
    const unsigned int m_lowrank;
    /** @brief sparse representation of matrix A_0*/
    SparseMatrixType A_sparse;
    /** @brief matrix B */
    DenseMatrixType B;
    /** @brief dense representation of m x m matrix Sigma^{-1} */
    DenseMatrixType Sigma_inv;
};

#endif // LINEAR_OPERATOR_HH
