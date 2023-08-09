#include "linear_operator.hh"
/** @file linear_operator.cc
 *
 * @brief Implementation of linear_operator.hh
 *
 * Defines the specific offsets for operators
 */

/* Coarsen linear operator to the next-coarser level*/
LinearOperator LinearOperator::coarsen(const std::shared_ptr<IntergridOperator> intergrid_operator) const
{
    const SparseMatrixType &A_restrict = intergrid_operator->to_sparse();
    const SparseMatrixType &A_prolong = A_restrict.transpose();
    // Galerkin triple product of sparse part
    const SparseMatrixType PT_A_P = A_restrict * A_sparse * A_prolong;
    LinearOperator lin_op(lattice->get_coarse_lattice(), m_lowrank);
    // Copy internal matrix representations
    lin_op.A_sparse = PT_A_P;
    lin_op.B = A_restrict * B;
    lin_op.Sigma_inv_diag = Sigma_inv_diag;
    lin_op.Sigma_inv_BT = Sigma_inv_diag * lin_op.B.transpose();
    return lin_op;
}

/* compute (dense) precision matrix */
LinearOperator::DenseMatrixType LinearOperator::precision() const
{
    DenseMatrixType Q = A_sparse.toDense();
    if (m_lowrank > 0)
    {
        Q += B * Sigma_inv_diag * B.transpose();
    }
    return Q;
}