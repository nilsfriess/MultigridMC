#ifndef INTERGRID_OPERATOR_LINEAR_HH
#define INTERGRID_OPERATOR_LINEAR_HH INTERGRID_OPERATOR_LINEAR_HH
#include <memory>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "intergrid_operator.hh"
#include "lattice/lattice.hh"

/** @file intergrid_operator_linear.hh
 * @brief Header file for linear interpolation intergrid operator
 */

/** @class IntergridOperatorLinear
 * IntergridOperator which implements linear averaging interpolation
 *
 */
class IntergridOperatorLinear : public IntergridOperator
{
public:
    /** @brief Base type */
    typedef IntergridOperator Base;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice object
     */
    IntergridOperatorLinear(const std::shared_ptr<Lattice> lattice_);
};

/* ******************** factory classes ****************************** */

/** @brief Factory for linear intergrid operators */
class IntergridOperatorLinearFactory : public IntergridOperatorFactory
{
public:
    /** @brief extract a smoother for a given action */
    virtual std::shared_ptr<IntergridOperator> get(std::shared_ptr<Lattice> lattice)
    {
        return std::make_shared<IntergridOperatorLinear>(lattice);
    };
};

#endif // INTERGRID_OPERATOR_LINEAR_HH