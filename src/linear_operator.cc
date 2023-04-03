#include "linear_operator.hh"
/** @file linear_operator.cc
 *
 * @brief Implementation of linear_operator.hh
 *
 * Defines the specific offsets for operators
 */

/** @brief Convert to sparse storage format */
const Eigen::SparseMatrix<double> AbstractLinearOperator::to_sparse() const
{
    unsigned int nrow = lattice->M;
    Eigen::SparseMatrix<double> A_sparse(nrow, nrow);
    size_t stencil_size = get_stencil().size();
    unsigned int nnz = stencil_size * nrow;
    A_sparse.reserve(nnz);
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplet_list;
    triplet_list.reserve(nnz);
    for (unsigned int ell = 0; ell < nrow; ++ell)
    {
        for (int k = 0; k < stencil_size; ++k)
        {
            unsigned int j = stencil_size * ell + k;
            triplet_list.push_back(T(ell, colidx[j], matrix[j]));
        }
    }
    A_sparse.setFromTriplets(triplet_list.begin(), triplet_list.end());
    return A_sparse;
}

/** @brief Offsets in x-direction for 5point operator in 2d */
template <>
const int BaseLinearOperator2d<5, LinearOperator2d5pt>::offset_x[5] = {0, 0, 0, -1, +1};
/** @brief Offsets in y-direction for 5point operator in 2d */
template <>
const int BaseLinearOperator2d<5, LinearOperator2d5pt>::offset_y[5] = {0, -1, +1, 0, 0};

/** @brief Create a new instance */
LinearOperator2d5pt::LinearOperator2d5pt(const std::shared_ptr<Lattice2d> lattice_, std::mt19937_64 &rng_) : Base(lattice_, rng_)
{
    for (unsigned int j = 0; j < ny; ++j)
    {
        for (unsigned int i = 0; i < nx; ++i)
        {
            for (int k = 1; k < stencil_size; ++k)
            {
                unsigned int ell = ((j + offset_y[k] + ny) % ny) * nx + ((i + offset_x[k] + nx) % nx);
                colidx[ell * stencil_size + k] = ell;
            }
        }
    }
}