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
    std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();
    for (unsigned int ell_coarse = 0; ell_coarse < coarse_lattice->M; ++ell_coarse)
    {
        Eigen::VectorXi idx_coarse = coarse_lattice->idx_linear2euclidean(ell_coarse);
        unsigned int ell = lattice->idx_euclidean2linear(2 * idx_coarse);
        for (int j = 0; j < stencil_size; ++j)
        {
            colidx[ell_coarse * stencil_size + j] = lattice->shift_index(ell, shift[j]);
        }
    }
}
