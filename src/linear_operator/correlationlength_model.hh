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
    typedef CorrelationLengthModel Base;
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

#endif // CORRELATIONLENGTH_MODEL_HH