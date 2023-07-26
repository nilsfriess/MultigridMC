#include "intergrid_operator.hh"

/** @file intergrid_operator.cc
 * @brief Implementation of intergrid_operator.hh
 */

/* Compute column indices on entire lattice */
void IntergridOperator::compute_colidx(const std::vector<Eigen::VectorXi> shift)
{
    std::shared_ptr<Lattice> coarse_lattice = lattice->get_coarse_lattice();
    for (unsigned int ell_coarse = 0; ell_coarse < coarse_lattice->Ncell; ++ell_coarse)
    {
        Eigen::VectorXi idx_coarse = coarse_lattice->cellidx_linear2euclidean(ell_coarse);
        unsigned int ell = lattice->cellidx_euclidean2linear(2 * idx_coarse);
        for (int j = 0; j < stencil_size; ++j)
        {
            colidx[ell_coarse * stencil_size + j] = lattice->shift_cellidx(ell, shift[j]);
        }
    }
}