#include "shiftedlaplace_fd_operator.hh"

/** @file shiftedlaplace_fd_operator.cc
 *
 * @brief Implementation of shiftedlaplace_fd_operator.hh
 */

/*  Create a new instance */
ShiftedLaplaceFDOperator::ShiftedLaplaceFDOperator(const std::shared_ptr<Lattice> lattice_,
                                                   const std::shared_ptr<CorrelationLengthModel> correlationlength_model_,
                                                   const int verbose) : LinearOperator(lattice_),
                                                                        correlationlength_model(correlationlength_model_)
{
    // dimension
    int dim = lattice->dim();
    // number of matrix rows
    int nrow = lattice->Nvertex;
    // shape of lattice
    Eigen::VectorXi shape = lattice->shape();
    // inverse squared grid spacings in all dimensions (required in 2nd order term)
    Eigen::VectorXd hinv2(dim);
    // cell volume
    double cell_volume = 1.0;
    for (int d = 0; d < dim; ++d)
    {
        double h = 1. / double(shape[d]);
        hinv2[d] = 1. / (h * h);
        cell_volume *= h;
    }
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve((1 + 2 * dim) * nrow);
    for (unsigned int ell = 0; ell < nrow; ++ell)
    {
        Eigen::VectorXd x = lattice->vertex_coordinates(ell);
        double diagonal = cell_volume * correlationlength_model->kappa_sq(x);
        for (int d = 0; d < dim; ++d) // loop over all dimensions
        {
            for (int j = 0; j < 2; ++j) // loop over all non-zero offsets
            {
                int offset = 2 * j - 1; // offset = -1, +1
                Eigen::VectorXi shift(dim);
                shift.setZero();
                shift[d] = offset;
                unsigned int ell_shifted;
                if (lattice->shifted_vertex_is_internal_vertex(ell, shift, ell_shifted))
                {
                    tripletList.push_back(T(ell, ell_shifted, -cell_volume * hinv2[d]));
                }
            }
            diagonal += 2. * cell_volume * hinv2[d];
        }
        // diagonal entry
        tripletList.push_back(T(ell, ell, diagonal));
    }
    A_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
}