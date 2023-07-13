#include "intergrid_operator_2dlinear.hh"

/** @file intergrid_operator_2dlinear.cc
 * @brief Implementation of intergrid_operator-2dlinear.hh
 */

/** @brief Create a new instance */
IntergridOperator2dLinear::IntergridOperator2dLinear(const std::shared_ptr<Lattice> lattice_) : Base(lattice_,
                                                                                                     int(pow(3, lattice_->dim())))
{
    int dim = lattice->dim();
    // 1d stencil and shift vector
    const double stencil1d[3] = {0.25, 0.5, 0.25};
    const int shift1d[3] = {-1, 0, +1};
    std::vector<Eigen::VectorXi> shift;
    // matrix entries and shifts
    for (int j = 0; j < stencil_size; ++j)
    {
        matrix[j] = 1.0;
        Eigen::VectorXi s(dim);
        int mu = j;
        for (int d = 0; d < dim; ++d)
        {
            matrix[j] *= stencil1d[mu % 3];
            s[d] = shift1d[mu % 3];
            mu /= 3;
        }
        shift.push_back(s);
    }
    compute_colidx(shift);
}