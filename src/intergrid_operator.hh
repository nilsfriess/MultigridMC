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

/** @class AbstractIntergridOperator
 *
 * @brief abstract IntergridOperator that can be used as a base class
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
class AbstractIntergridOperator
{
public:
    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice
     */
    AbstractIntergridOperator(const std::shared_ptr<Lattice> lattice_) : lattice(lattice_) {}

    /** @brief Get the matrix entries of the prolongation operator */
    virtual const double *get_matrix() const = 0;

    /** @brief Restrict a vector to the next-coarser level
     *
     * Compute x^{c} = I_{h}^{2h} x
     *
     * @param[in] x input vector on current lattice
     * @param[out] x^{c} output vector on next-coarser lattice
     */
    virtual void restrict(const std::shared_ptr<SampleState> x, std::shared_ptr<SampleState> x_coarse) const = 0;

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
    virtual void prolongate_add(const std::shared_ptr<SampleState> x_coarse, std::shared_ptr<SampleState> x) const = 0;

    /** @brief Coarsen a linear operator to the next-coarser level
     *
     * Compute A^{c} = I_{2h}^{h} A I_{h}^{2h}
     *
     * To compute A^{(c)} note that we need the matrix entries
     *
     *   A^{c}_{i,i+\Delta_k}
     *
     * for all coarse grid points i and all offsets \Delta_k in S(A^{c}), the set of possible
     * offsets of the coarse level operator A^{c}. This can be written as
     *
     *   A^{c}_{i,i+\Delta_k} = \sum{\sigma^{k}_m,\sigma'^{k}_m} in S(A;\Delta_k)
     *                            P_{\sigma^{k}_m} * P_{\sigma'^{k}_m}
     *                            * A_{2*i+\sigma^{k}_m,2*(i+\Delta_k)+\sigma'^{k}_m}
     *
     * where for each offset \Delta_k the set S(A;\Delta_k) contains all pairs of
     * offsets (\sigma^{k}_m,\sigma'^{k}_m) such that 2*\Delta_k + \sigma'^{k}_m -  \sigma^{k}_m in S(A),
     * the set of possible offsets of the operator A.
     *
     * @param[in] A Linear operator on current lattice
     * @param[out] A_coarse Resulting linear operator on current lattice
     */
    virtual void coarsen_operator(const std::shared_ptr<AbstractLinearOperator> A, std::shared_ptr<AbstractLinearOperator> A_coarse) const;

    /** @extract the stencil */
    virtual std::vector<Eigen::VectorXi> get_stencil() const = 0;

    /** @brief underlying lattice */
    const std::shared_ptr<Lattice> lattice;
};

/** @class BaseIntergridOperator2d
 *
 * @brief Base class for defining IntergridOperators with the CRTP in 2d
 *
 * Intergrid operator that allows describing restriction, prolongation and matrix-coarsening
 * for regular 2d lattices.
 *
 */
template <int ssize, class DerivedIntergridOperator>
class BaseIntergridOperator2d : public AbstractIntergridOperator
{
public:
    typedef BaseIntergridOperator2d<ssize, DerivedIntergridOperator> Base;
    /** @brief Stencil size */
    static const int stencil_size = ssize;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying 2d lattice
     */
    BaseIntergridOperator2d(const std::shared_ptr<Lattice2d> lattice_) : AbstractIntergridOperator(lattice_), nx(lattice_->nx), ny(lattice_->ny)
    {
        colidx = new unsigned int[lattice_->M * ssize];
    }

    /** @brief Destroy instance */
    BaseIntergridOperator2d()
    {
        delete[] colidx;
    }

    /** @brief Get the matrix entries of the prolongation operator */
    virtual const double *get_matrix() const
    {
        return matrix;
    };

    /** @brief Get the offsets of the prolongation operator */
    virtual std::vector<Eigen::VectorXi> get_stencil() const
    {
        std::vector<Eigen::VectorXi> offsets;
        for (int j = 0; j < stencil_size; ++j)
        {
            Eigen::VectorXi v(2);
            v[0] = offset_x[j];
            v[1] = offset_y[j];
            offsets.push_back(v);
        }
        return offsets;
    };

    /** @brief Restrict a vector to the next-coarser level
     *
     * Compute x^{c} = I_{h}^{2h} x
     *
     * @param[in] x input vector on current lattice
     * @param[out] x^{c} output vector on next-coarser lattice
     */
    virtual void restrict(const std::shared_ptr<SampleState> x, std::shared_ptr<SampleState> x_coarse) const
    {
        for (unsigned int j = 0; j < ny / 2; ++j)
        {
            for (unsigned int i = 0; i < nx / 2; ++i)
            {
                unsigned int ell_coarse = j * (nx / 2) + i;
                double result = 0;
                for (unsigned k = 0; k < ssize; ++k)
                {
                    unsigned int ell = ((2 * j + offset_y[k]) % ny) * nx + ((2 * i + offset_x[k]) % nx);
                    result += matrix[ell * ssize + k] * x->data[ell];
                }
                x_coarse->data[ell_coarse] = result;
            }
        }
    };

    /** @brief Prolongate-add a vector from the next-coarser level
     *
     * Compute x += I_{2h}^{h} x^{c}
     *
     * @param[in] x_coarse input vector on next-coarser lattice
     * @param[out] x output vector on current lattice
     */
    virtual void prolongate_add(const std::shared_ptr<SampleState> x_coarse, std::shared_ptr<SampleState> x) const
    {
        for (unsigned int j = 0; j < ny / 2; ++j)
        {
            for (unsigned int i = 0; i < nx / 2; ++i)
            {
                unsigned int ell_coarse = j * (nx / 2) + i;
                double x_coarse_local = x_coarse->data[ell_coarse];
                for (unsigned k = 0; k < ssize; ++k)
                {
                    unsigned int ell = ((2 * j + offset_y[k]) % ny) * nx + ((2 * i + offset_x[k]) % nx);
                    x->data[ell] += matrix[ell * ssize + k] * x_coarse_local;
                }
            }
        }
    };

protected:
    /** @brief extent of lattice in x-direction */
    const unsigned int nx;
    /** @brief extent of lattice in y-direction */
    const unsigned int ny;
    /** @brief matrix entries */
    static const double matrix[ssize];
    /** @brief Offsets in x-direction */
    static const int offset_x[ssize];
    /** @brief Offsets in y-direction */
    static const int offset_y[ssize];
    /** @brief column indices */
    unsigned int *colidx;
};

/** @class IntergridOperatorAvg
 * IntergridOperator which implements constant averaging
 *
 */
class IntergridOperator2dAvg : public BaseIntergridOperator2d<4, IntergridOperator2dAvg>
{
public:
    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice object
     */
    IntergridOperator2dAvg(const std::shared_ptr<Lattice2d> lattice_);
};

#endif // INTERGRID_OPERATOR_HH