#ifndef OPERATOR_HH
#define OPERATOR_HH OPERATOR_HH
#include <memory>
#include <random>
#include <cmath>
#include <Eigen/Dense>
#include "lattice2d.hh"
#include "samplestate.hh"

/** @file operator.hh
 * @brief Header file for linear operator classes
 */

/** @class AbstractOperator
 *
 * @brief abstract linear operator that can be used as a base class
 */
class AbstractOperator
{
public:
    /** @brief Create a new instance
     *
     * @param[in] rng random number generator
     */
    AbstractOperator(std::mt19937_64 &rng_) : rng(rng_), normal_dist(0.0, 1.0) {}

    /** @brief Apply the linear operator
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

/** @class BaseOperator2d
 *
 * @brief base class for defining operators with the CRTP in 2d
 */
template <int ssize, class DerivedOperator>
class BaseOperator2d : public AbstractOperator
{
public:
    typedef BaseOperator2d<ssize, DerivedOperator> Base;
    /** @brief Stencil size */
    static const int stencil_size = ssize;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying 2d lattice
     * @param[in] rng_ random number generator
     */
    BaseOperator2d(const Lattice2d &lattice_, std::mt19937_64 &rng_) : lattice(lattice_), AbstractOperator(rng_)
    {
        data = new double[ssize * lattice.M];
    }

    /**@brief Destrory instance*/
    ~BaseOperator2d()
    {
        delete[] data;
    }

    /** @brief Apply the linear operator
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
            stencil.push_back(Eigen::VectorXi(offset_x[j], offset_y[j]));
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

/** @class Operator2d5pt
 * Operator with 5-point stencil. The stencil elements are arrange in the following order, with
 * offsets in x- and y-direction shown in brackets
 *
 * 0 : centre (  0,  0)
 * 1 : south  (  0, -1)
 * 2 : north  (  0, +1)
 * 3 : east   ( -1,  0)
 * 4 : west   ( +1,  0)
 */
class Operator2d5pt : public BaseOperator2d<5, Operator2d5pt>
{
public:
    /** @brief Create a new instance 
     * 
     * @param[in] lattice_ underlying lattice object
     * @param[in] rng_ random number generator (for Gibbs sweep)
    */
    Operator2d5pt(const Lattice2d &lattice_, std::mt19937_64 &rng_) : Base(lattice_, rng_) {}
};

/** @brief Offsets in x-direction for 5point operator in 2d */
template <>
const int BaseOperator2d<5, Operator2d5pt>::offset_x[5] = {0, 0, 0, -1, +1};
/** @brief Offsets in y-direction for 5point operator in 2d */
template <>
const int BaseOperator2d<5, Operator2d5pt>::offset_y[5] = {0, -1, +1, 0, 0};

#endif // OPERATOR_HH