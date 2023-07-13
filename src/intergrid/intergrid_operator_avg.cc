#include "intergrid_operator_avg.hh"

/** @file intergrid_operator_avg.cc
 * @brief Implementation of intergrid_operator_avg.hh
 */

/** @brief Create a new instance */
IntergridOperatorAvg::IntergridOperatorAvg(const std::shared_ptr<Lattice> lattice_) : Base(lattice_,
                                                                                           1 << lattice_->dim())
{
    unsigned int dim = lattice->dim();
    // set matrix entries and shifts
    std::vector<Eigen::VectorXi> shift;
    for (int j = 0; j < stencil_size; ++j)
    {
        matrix[j] = 1.0;
        Eigen::VectorXi s(dim);
        for (int k = 0; k < dim; ++k)
            s[k] = 1 & (j >> k);
        shift.push_back(s);
    }
    // column indices
    compute_colidx(shift);
}
