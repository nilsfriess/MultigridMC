#ifndef INTERGRID_OPERATOR_HH
#define INTERGRID_OPERATOR_HH INTERGRID_OPERATOR_HH
#include <memory>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include "lattice.hh"
#include "linear_operator.hh"
#include "samplestate.hh"

/** @file intergrid_operator.hh
 * @brief Header file for intergrid operator classes
 */

/** @class IntergridOperator
 *
 * @brief IntergridOperator that can be used as a base class
 *
 * A linear operator class provides functionality for performing the
 * following operations:
 *
 *  - Restrict a vector on a lattice to a vector on the next-coarser lattice
 *
 *      x^{c} = I_{h}^{2h} x
 *
 *  - Prolongate-add a vector from the next-coarser level
 *
 *      x += I_{2h}^{h} x^{c}
 *
 *  - Coarsen a linear operator to the next-coarser lattice using the
 *    Galerkin triple-product
 *
 *       A^{c} = I_{h}^{2h} A I_{2h}^{h}
 *
 * It is assumed that I_{h}^{2h} = (I_{2h}^{h})^T.
 * The sparsity structure of the intergrid operator is assumed to be fixed and is described
 * by a stencil; the entries of the matrices I_{h}^{2h} and I_{2h}^{h} are also assumed to
 * be constant. Note in particular that this implies that each coarse level unknown depends on
 * the fine level unknown in exactly the same way.
 */
class IntergridOperator
{
public:
    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice
     * @param[in] stencil_size_ size of intergrid stencil
     */
    IntergridOperator(const std::shared_ptr<Lattice> lattice_,
                      const int stencil_size_) : lattice(lattice_),
                                                 stencil_size(stencil_size_)
    {
        matrix = new double[stencil_size];
        colidx = new unsigned int[lattice->M * stencil_size];
    }

    ~IntergridOperator()
    {
        delete[] matrix;
        delete[] colidx;
    }

    /** @brief Restrict a vector to the next-coarser level
     *
     * Compute x^{c} = I_{h}^{2h} x
     *
     * @param[in] x input vector on current lattice
     * @param[out] x^{c} output vector on next-coarser lattice
     */
    virtual void restrict(const std::shared_ptr<SampleState> x, std::shared_ptr<SampleState> x_coarse)
    {
        std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();
        for (unsigned int ell_coarse = 0; ell_coarse < coarse_lattice->M; ++ell_coarse)
        {
            double result = 0;
            for (unsigned k = 0; k < stencil_size; ++k)
            {
                unsigned int ell = colidx[ell_coarse * stencil_size + k];
                result += matrix[k] * x->data[ell];
            }
            x_coarse->data[ell_coarse] = result;
        }
    }

    /** @brief Prolongate-add a vector from the next-coarser level
     *
     * Compute x += I_{2h}^{h} x^{c}
     *
     * The prolongation operator does not depend on position, so the operation can be written as:
     *
     * x_i  = x_i + \sum_j (I_{2h}^h)_{ij} x^{c}_j
     *      = x_i + \sum_{\sigma} (I_{2h}^h)_{i,2*i+\sigma} x^{c}_{2*i+\sigma}
     *      = x_i + \sum_{\sigma} P_{\sigma} x^{c}_{2*i+\sigma}
     *
     * where \sigma runs over all possible offsets.
     *
     * @param[in] x_coarse input vector on next-coarser lattice
     * @param[out] x output vector on current lattice
     */
    virtual void prolongate_add(const std::shared_ptr<SampleState> x_coarse, std::shared_ptr<SampleState> x)
    {
        std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();
        for (unsigned int ell_coarse = 0; ell_coarse < coarse_lattice->M; ++ell_coarse)
        {
            double x_coarse_ell = x_coarse->data[ell_coarse];
            for (unsigned k = 0; k < stencil_size; ++k)
            {
                unsigned int ell = colidx[ell_coarse * stencil_size + k];
                x->data[ell] += matrix[k] * x_coarse_ell;
            }
        }
    };

    /** @brief convert prolongation operator to a sparse matrix */
    const Eigen::SparseMatrix<double> to_sparse() const
    {
        std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();
        typedef Eigen::Triplet<double> T;
        std::vector<T> triplet_list;
        unsigned int nrow = lattice->M;
        unsigned int ncol = coarse_lattice->M;
        unsigned int nnz = stencil_size * nrow;
        triplet_list.reserve(nnz);
        for (unsigned int ell = 0; ell < ncol; ++ell)
        {
            {
                for (int k = 0; k < stencil_size; ++k)
                {
                    triplet_list.push_back(T(colidx[ell * stencil_size + k], ell, matrix[k]));
                }
            }
        }
        LinearOperator::SparseMatrixType A_sparse(nrow, ncol);
        A_sparse.setFromTriplets(triplet_list.begin(), triplet_list.end());
        return A_sparse;
    }

    /** @brief Coarsen a linear operator to the next-coarser level
     *
     * Compute A^{c} = I_{2h}^{h} A I_{h}^{2h}
     *
     * @param[in] A Linear operator on current lattice
     */
    LinearOperator coarsen_operator(const LinearOperator &A) const
    {
        const LinearOperator::SparseMatrixType &A_prolong = to_sparse();
        const LinearOperator::SparseMatrixType PT_A_P = A_prolong.transpose() * A.to_sparse() * A_prolong;
        return LinearOperator(lattice->get_coarse_lattice(), PT_A_P);
    }

    /** @brief underlying lattice */
    const std::shared_ptr<Lattice> lattice;
    /** @brief size of stencil */
    const int stencil_size;
    /** @brief underlying matrix */
    double *matrix;
    /** @brief indirection map */
    unsigned int *colidx;
};

/** @class IntergridOperatorAvg
 * IntergridOperator which implements constant averaging
 *
 */
class IntergridOperator2dAvg : public IntergridOperator
{
public:
    /** @brief Base type */
    typedef IntergridOperator Base;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice object
     */
    IntergridOperator2dAvg(const std::shared_ptr<Lattice2d> lattice_);
};

#endif // INTERGRID_OPERATOR_HH