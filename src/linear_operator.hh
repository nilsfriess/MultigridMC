#ifndef LINEAR_OPERATOR_HH
#define LINEAR_OPERATOR_HH LINEAR_OPERATOR_HH
#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include "lattice.hh"
#include "samplestate.hh"

/** @file LinearOperator.hh
 * @brief Header file for LinearOperator classes
 */

/** @class AbstractLinearOperator
 *
 * @brief abstract linear LinearOperator that can be used as a base class
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
class AbstractLinearOperator
{
public:
    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice
     * @param[in] rng random number generator
     */
    AbstractLinearOperator(const std::shared_ptr<Lattice> lattice_, std::mt19937_64 &rng_) : lattice(lattice_), rng(rng_), normal_dist(0.0, 1.0) {}

    /** @brief Extract underlying lattice */
    std::shared_ptr<Lattice> get_lattice() const { return lattice; }

    /** @brief Extract pointer to matrix elements */
    virtual double *get_matrix() const { return matrix; }

    /** @brief Apply the linear LinearOperator
     *
     * Compute y = Ax
     *
     * @param[in] x input vector
     * @param[out] y output vector
     */
    virtual void apply(const std::shared_ptr<SampleState> x, std::shared_ptr<SampleState> y) const = 0;

    /** @brief Carry out a single Gibbs-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void gibbssweep(const std::shared_ptr<SampleState> x, std::shared_ptr<SampleState> y) = 0;

    /** @extract the stencil */
    virtual std::vector<Eigen::VectorXi> get_stencil() const = 0;

protected:
    /** @brief underlying lattice */
    const std::shared_ptr<Lattice> lattice;
    /** @brief random number generator */
    std::mt19937_64 &rng;
    /** @brief normal distribution for Gibbs-sweep */
    std::normal_distribution<double> normal_dist;
    /** @brief matrix entries */
    double *matrix;
};

/** @class BaseLinearOperator2d
 *
 * @brief base class for defining LinearOperators with the CRTP in 2d
 *
 * Linear operator for vectors that describe the unknowns on a structured 2d lattice.
 * The unknowns are assumed to be arranged in lexicographic order, i.e. node (i,j)
 * maps to the index ell = nx*j + i. In this case the sparsity structure can be described
 * by a list of offsets in x- and y-direction.
 *
 */
template <int ssize, class DerivedLinearOperator>
class BaseLinearOperator2d : public AbstractLinearOperator
{
public:
    typedef BaseLinearOperator2d<ssize, DerivedLinearOperator> Base;
    /** @brief Stencil size */
    static const int stencil_size = ssize;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying 2d lattice
     * @param[in] rng_ random number generator
     */
    BaseLinearOperator2d(const std::shared_ptr<Lattice2d> lattice_, std::mt19937_64 &rng_) : AbstractLinearOperator(lattice_, rng_), nx(lattice_->nx), ny(lattice_->ny)
    {
        matrix = new double[ssize * lattice->M];
    }

    /**@brief Destrory instance*/
    ~BaseLinearOperator2d()
    {
        delete[] matrix;
    }

    /** @brief Extract underlying lattice */
    std::shared_ptr<Lattice> get_lattice() const { return lattice; }

    /** @brief Apply the linear LinearOperator
     *
     * Compute y = Ax
     *
     * @param[in] x input vector
     * @param[out] y output vector
     */
    virtual void apply(const std::shared_ptr<SampleState> x, std::shared_ptr<SampleState> y) const
    {
        for (unsigned int j = 0; j < ny; ++j)
        {
            for (unsigned int i = 0; i < nx; ++i)
            {
                unsigned int ell = j * nx + i;
                double result = 0;
                for (unsigned k = 0; k < ssize; ++k)
                {
                    unsigned int ell_prime = ((j + offset_y[k]) % ny) * nx + ((i + offset_x[k]) % nx);
                    result += matrix[ell * ssize + k] * x->data[ell_prime];
                }
                y->data[ell] = result;
            }
        }
    }

    /** @brief Carry out a single Gibbs-sweep
     *
     * @param[in] b right hand side
     * @param[inout] x vector to which the sweep is applied
     */
    virtual void gibbssweep(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> x)
    {
        for (unsigned int j = 0; j < ny; ++j)
        {
            for (unsigned int i = 0; i < nx; ++i)
            {
                unsigned int ell = j * nx + i;
                double residual = 0;
                double a_diag = matrix[ell * ssize];
                for (unsigned k = 1; k < ssize; ++k)
                {
                    unsigned int ell_prime = ((j + offset_y[k]) % ny) * nx + ((i + offset_x[k]) % nx);
                    residual += matrix[ell * ssize + k] * x->data[ell_prime];
                }
                x->data[ell] = (b->data[ell] - residual) / a_diag + normal_dist(rng) / sqrt(a_diag);
            }
        }
    }

    /** @brief extract the stencil as a vector of offsets */
    virtual std::vector<Eigen::VectorXi> get_stencil() const
    {
        std::vector<Eigen::VectorXi> stencil;
        for (int j = 0; j < stencil_size; ++j)
        {
            Eigen::VectorXi v(2);
            v[0] = offset_x[j];
            v[1] = offset_y[j];
            stencil.push_back(v);
        }
        return stencil;
    };

protected:
    /** @brief lattice extent in x-direction */
    const unsigned int nx;
    /** @brief lattice extent in y-direction */
    const unsigned int ny;
    /** @brief Offsets in x-direction */
    static const int offset_x[ssize];
    /** @brief Offsets in y-direction */
    static const int offset_y[ssize];
};

/** @class LinearOperator2d5pt
 * LinearOperator with 5-point stencil. The stencil elements are arrange in the following order, with
 * offsets in x- and y-direction shown in brackets
 *
 * 0 : centre (  0,  0)
 * 1 : south  (  0, -1)
 * 2 : north  (  0, +1)
 * 3 : east   ( -1,  0)
 * 4 : west   ( +1,  0)
 */
class LinearOperator2d5pt : public BaseLinearOperator2d<5, LinearOperator2d5pt>
{
public:
    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice object
     * @param[in] rng_ random number generator (for Gibbs sweep)
     */
    LinearOperator2d5pt(const std::shared_ptr<Lattice2d> lattice_, std::mt19937_64 &rng_) : Base(lattice_, rng_) {}
};

#endif // LINEAR_OPERATOR_HH
