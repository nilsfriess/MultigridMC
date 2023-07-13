#ifndef INTERGRID_OPERATOR_AVG_HH
#define INTERGRID_OPERATOR_AVG_HH INTERGRID_OPERATOR_AVG_HH
#include <memory>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "lattice/lattice.hh"
#include "intergrid_operator.hh"

/** @file intergrid_operator_avg.hh
 * @brief Header file for averaging intergrid operator
 */

/** @class IntergridOperatorAvg
 * IntergridOperator which implements constant averaging
 *
 */
class IntergridOperatorAvg : public IntergridOperator
{
public:
    /** @brief Base type */
    typedef IntergridOperator Base;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice object
     */
    IntergridOperatorAvg(const std::shared_ptr<Lattice> lattice_);
};
/* ******************** factory classes ****************************** */

/** @brief Factory for averaging intergrid operators */
class IntergridOperatorAvgFactory : public IntergridOperatorFactory
{
public:
    /** @brief extract a smoother for a given action */
    virtual std::shared_ptr<IntergridOperator> get(std::shared_ptr<Lattice> lattice)
    {
        return std::make_shared<IntergridOperatorAvg>(lattice);
    };
};

#endif // INTERGRID_OPERATOR_AVG_HH