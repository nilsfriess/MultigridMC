#ifndef INTERGRID_OPERATOR_2DAVG_HH
#define INTERGRID_OPERATOR_2DAVG_HH INTERGRID_OPERATOR_2DAVG_HH
#include <memory>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "lattice/lattice2d.hh"
#include "intergrid_operator.hh"

/** @file intergrid_operator_2davg.hh
 * @brief Header file for 2d averaging intergrid operator
 */

/** @class IntergridOperator2dAvg
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
/* ******************** factory classes ****************************** */

/** @brief Factory for 2d averaging intergrid operators */
class IntergridOperator2dAvgFactory : public IntergridOperatorFactory
{
public:
    /** @brief extract a smoother for a given action */
    virtual std::shared_ptr<IntergridOperator> get(std::shared_ptr<Lattice> lattice)
    {
        return std::make_shared<IntergridOperator2dAvg>(std::dynamic_pointer_cast<Lattice2d>(lattice));
    };
};

#endif // INTERGRID_OPERATOR_2DAVG_HH