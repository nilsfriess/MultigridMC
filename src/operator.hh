#ifndef LinearOperator_HH
#define LinearOperator_HH LinearOperator_HH
#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include "lattice2d.hh"
#include "samplestate.hh"

/** @file LinearOperator.hh
 * @brief Header file for linear LinearOperator classes
 */

/** @class AbstractLinearOperator
 *
 * @brief abstract linear LinearOperator that can be used as a base class
 */
class AbstractLinearOperator
{
public:
    /** @brief Create a new instance
     *
     * @param[in] rng random number generator
     */
    AbstractLinearOperator(std::mt19937_64 &rng_) : rng(rng_), normal_dist(0.0, 1.0) {}

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
    /** @brief random number generator */
    std::mt19937_64 &rng;
    /** @brief normal distribution for Gibbs-sweep */
    std::normal_distribution<double> normal_dist;
};

/** @class BaseLinearOperator2d
 *
 * @brief base class for defining LinearOperators with the CRTP in 2d
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
    BaseLinearOperator2d(const Lattice2d &lattice_, std::mt19937_64 &rng_) : lattice(lattice_), AbstractLinearOperator(rng_)
    {
        data = new double[ssize * lattice.M];
    }

    /**@brief Destrory instance*/
    ~BaseLinearOperator2d()
    {
        delete[] data;
    }

    /** @brief Apply the linear LinearOperator
     *
     * Compute y = Ax
     *
     * @param[in] x input vector
     * @param[out] y output vector
     */
    virtual void apply(const std::shared_ptr<SampleState> x, std::shared_ptr<SampleState> y) const
    {
        unsigned int nx = lattice.nx;
        unsigned int ny = lattice.ny;
        for (unsigned int j = 0; j < ny; ++j)
        {
            for (unsigned int i = 0; i < nx; ++i)
            {
                unsigned int ell = j * nx + i;
                double result = 0;
                for (unsigned k = 0; k < ssize; ++k)
                {
                    unsigned int ell_prime = ((j + offset_y[k]) % ny) * nx + ((i + offset_x[k]) % nx);
                    result += data[ell * ssize + k] * x->data[ell_prime];
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
        unsigned int nx = lattice.nx;
        unsigned int ny = lattice.ny;
        for (unsigned int j = 0; j < ny; ++j)
        {
            for (unsigned int i = 0; i < nx; ++i)
            {
                unsigned int ell = j * nx + i;
                double residual = 0;
                double a_diag = data[ell * ssize];
                for (unsigned k = 1; k < ssize; ++k)
                {
                    unsigned int ell_prime = ((j + offset_y[k]) % ny) * nx + ((i + offset_x[k]) % nx);
                    residual += data[ell * ssize + k] * x->data[ell_prime];
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
    /** @brief underlying lattice */
    const Lattice2d &lattice;
    /** @brief matrix entries */
    double *data;
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
    LinearOperator2d5pt(const Lattice2d &lattice_, std::mt19937_64 &rng_) : Base(lattice_, rng_) {}
};

/** @brief Offsets in x-direction for 5point LinearOperator in 2d */
template <>
const int BaseLinearOperator2d<5, LinearOperator2d5pt>::offset_x[5] = {0, 0, 0, -1, +1};
/** @brief Offsets in y-direction for 5point LinearOperator in 2d */
template <>
const int BaseLinearOperator2d<5, LinearOperator2d5pt>::offset_y[5] = {0, -1, +1, 0, 0};

#endif // LinearOperator_HH