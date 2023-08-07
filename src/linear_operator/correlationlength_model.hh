#ifndef CORRELATIONLENGTH_MODEL_HH
#define CORRELATIONLENGTH_MODEL_HH CORRELATIONLENGTH_MODEL_HH

#include <memory>
#include <cmath>
#include <Eigen/Dense>
#include "auxilliary/parameters.hh"
#include "lattice/lattice.hh"

/** @file correlationlength_model.hh
 *
 * @brief Models for correlation length in computation domain
 */

/** @class CorrelationLengthModel
 *
 * @brief virtual base class of all models
 */

class CorrelationLengthModel
{
public:
    /** @brief compute correlation length at a particular point in the domain
     *
     * @param[in] x point at which to evaluate the correlation length
     */
    double kappa(const Eigen::VectorXd x) const { return 1. / sqrt(kappa_invsq(x)); }

    /** @brief compute inverse squared correlation length
     *
     * @param[in] x point at which to evaluate the correlation length
     */
    inline virtual double kappa_invsq(const Eigen::VectorXd x) const = 0;

protected:
    /** @brief Underlying lattice */
    const std::shared_ptr<Lattice> lattice;
};

/** @class ConstantCorrelationLengthModel
 *
 * @brief Model for constant correlation length in the entire domain
 */

class ConstantCorrelationLengthModel : public CorrelationLengthModel
{
public:
    /** @brief Constructor
     *
     * @param[in] params_ parameters
     */
    ConstantCorrelationLengthModel(const ConstantCorrelationLengthModelParameters params_) : kappa_invsq_(1. / pow(params_.kappa, 2)) {}

    /** @brief compute correlation length at a particular point in the domain
     *
     * @param[in] x point at which to evaluate the correlation length
     */
    inline virtual double kappa_invsq(const Eigen::VectorXd x) const
    {
        return kappa_invsq_;
    }

protected:
    /** @brief Inverse squared correlation length */
    const double kappa_invsq_;
};

/** @class PeriodicCorrelationLengthModel
 *
 * @brief Model for correlation length that varies periodically
 *
 * The correlation length is assumed to be of the form
 *
 *    kappa(x_1,x_2,...,x_d) = kappa_1 + kappa_2 * cos(pi*x_1) * cos(pi*x_2) * ... * cos(pi*x_d)
 *
 * where
 *
 *    kappa_1 = 1/2(kappa_max + kappa_min)
 *
 *    kappa_2 = 1/2(kappa_max - kappa_min)
 */

class PeriodicCorrelationLengthModel : public CorrelationLengthModel
{
public:
    /** @brief Constructor
     *
     * @param[in] params_ parameters
     */
    PeriodicCorrelationLengthModel(const PeriodicCorrelationLengthModelParameters params_) : kappa_1(0.5 * (params_.kappa_max + params_.kappa_min)),
                                                                                             kappa_2(0.5 * (params_.kappa_max - params_.kappa_min)) {}

    /** @brief compute correlation length at a particular point in the domain
     *
     * @param[in] x point at which to evaluate the correlation length
     */
    inline virtual double kappa_invsq(const Eigen::VectorXd x) const
    {
        int dim = x.size();
        double kappa_ = kappa_2;
        for (int d = 0; d < dim; ++d)
            kappa_ *= cos(M_PI * x[d]);
        kappa_ += kappa_1;
        return 1. / (kappa_ * kappa_);
    }

protected:
    /** @brief parameter kappa_1 = 1/2 * (kappa_max + kappa_min) */
    const double kappa_1;
    /** @brief parameter kappa_2 = 1/2 * (kappa_max - kappa_min) */
    const double kappa_2;
};

#endif // CORRELATIONLENGTH_MODEL_HH