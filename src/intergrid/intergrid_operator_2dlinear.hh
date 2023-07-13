#ifndef INTERGRID_OPERATOR_2DLINEAR_HH
#define INTERGRID_OPERATOR_2DLINEAR_HH INTERGRID_OPERATOR_2DLINEAR_HH
#include <memory>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "intergrid_operator.hh"
#include "lattice/lattice2d.hh"

/** @file intergrid_operator.hh
 * @brief Header file for 2d linear interpolation intergrid operator
 */

/** @class IntergridOperator2dLinear
 * IntergridOperator which implements linear averaging interpolation
 *
 */
class IntergridOperator2dLinear : public IntergridOperator
{
public:
    /** @brief Base type */
    typedef IntergridOperator Base;

    /** @brief Create a new instance
     *
     * @param[in] lattice_ underlying lattice object
     */
    IntergridOperator2dLinear(const std::shared_ptr<Lattice> lattice_);

protected:
    /** @brief Compute column indices on entire lattice
     *
     * @param[in] shift vector with shift indices
     */
    void compute_colidx(const std::vector<Eigen::VectorXi> shift);
};

/* ******************** factory classes ****************************** */

/** @brief Factory for 2d linear intergrid operators */
class IntergridOperator2dLinearFactory : public IntergridOperatorFactory
{
public:
    /** @brief extract a smoother for a given action */
    virtual std::shared_ptr<IntergridOperator> get(std::shared_ptr<Lattice> lattice)
    {
        return std::make_shared<IntergridOperator2dLinear>(lattice);
    };
};

#endif // INTERGRID_OPERATOR_2DLINEAR_HH