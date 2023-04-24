#ifndef INTERGRID_OPERATOR_1DLINEAR_HH
#define INTERGRID_OPERATOR_1DLINEAR_HH INTERGRID_OPERATOR_1DLINEAR_HH
#include <memory>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "intergrid_operator.hh"
#include "lattice/lattice1d.hh"

/** @file intergrid_operator_1dlinear.hh
 * @brief Header file for 1d linear interpolation intergrid operator
 */

/** @class IntergridOperator1dLinear
 * IntergridOperator which implements linear averaging interpolation
 *
 */
class IntergridOperator1dLinear : public IntergridOperator
{
public:
    /** @brief Base type */
    typedef IntergridOperator Base;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice object
     */
    IntergridOperator1dLinear(const std::shared_ptr<Lattice1d> lattice_);
};

/* ******************** factory classes ****************************** */

/** @brief Factory for 1d linear intergrid operators */
class IntergridOperator1dLinearFactory : public IntergridOperatorFactory
{
public:
    /** @brief extract a smoother for a given action */
    virtual std::shared_ptr<IntergridOperator> get(std::shared_ptr<Lattice> lattice)
    {
        return std::make_shared<IntergridOperator1dLinear>(std::dynamic_pointer_cast<Lattice1d>(lattice));
    };
};

#endif // INTERGRID_OPERATOR_1DLINEAR_HH