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
    double kappa(const Eigen::VectorXd x) const { return 1. / sqrt(kappa_sq(x)); }

    /** @brief compute inverse squared correlation length
     *
     * @param[in] x point at which to evaluate the correlation length
     */
    inline virtual double kappa_sq(const Eigen::VectorXd x) const = 0;

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
    ConstantCorrelationLengthModel(const ConstantCorrelationLengthModelParameters params_) : kappa_sq_(1. / pow(params_.Lambda, 2)) {}

    /** @brief compute squared correlation length at a particular point in the domain
     *
     * @param[in] x point at which to evaluate the correlation length
     */
    inline virtual double kappa_sq(const Eigen::VectorXd x) const
    {
        return kappa_sq_;
    }

protected:
    /** @brief Inverse squared correlation length */
    const double kappa_sq_;
};

/** @class PeriodicCorrelationLengthModel
 *
 * @brief Model for correlation length that varies periodically
 *
 * The correlation length is assumed to be of the form
 *
 *    Lambda(x_1,x_2,...,x_d) = Lambda_1 + Lambda_2 * cos(pi*x_1) * cos(pi*x_2) * ... * cos(pi*x_d)
 *
 * where
 *
 *    Lambda_1 = 1/2 * (Lambda_max + Lambda_min)
 *
 *    Lambda_2 = 1/2 * (Lambda_max - Lambda_min)
 */

class PeriodicCorrelationLengthModel : public CorrelationLengthModel
{
public:
    /** @brief Constructor
     *
     * @param[in] params_ parameters
     */
    PeriodicCorrelationLengthModel(const PeriodicCorrelationLengthModelParameters params_) : Lambda_1(0.5 * (params_.Lambda_max + params_.Lambda_min)),
                                                                                             Lambda_2(0.5 * (params_.Lambda_max - params_.Lambda_min)) {}

    /** @brief compute inverse squadr correlation length at a particular point in the domain
     *
     * @param[in] x point at which to evaluate the correlation length
     */
    inline virtual double kappa_sq(const Eigen::VectorXd x) const
    {
        int dim = x.size();
        double Lambda_ = Lambda_2;
        for (int d = 0; d < dim; ++d)
            Lambda_ *= cos(M_PI * x[d]);
        Lambda_ += Lambda_1;
        return 1. / (Lambda_ * Lambda_);
    }

protected:
    /** @brief parameter Lambda_1 = 1/2 * (Lambda_max + Lambda_min) */
    const double Lambda_1;
    /** @brief parameter Lambda_2 = 1/2 * (Lambda_max - Lambda_min) */
    const double Lambda_2;
};

#endif // CORRELATIONLENGTH_MODEL_HH