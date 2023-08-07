#include "squared_shiftedlaplace_fd_operator.hh"

/** @file squared_shiftedlaplace_fd_operator.cc
 *
 * @brief Implementation of shiftedbiharmonic_operator.hh
 */

/*  Create a new instance */
SquaredShiftedLaplaceFDOperator::SquaredShiftedLaplaceFDOperator(const std::shared_ptr<Lattice> lattice_,
                                                                 const std::shared_ptr<CorrelationLengthModel> correlationlength_model_,
                                                                 const int verbose) : LinearOperator(lattice_),
                                                                                      correlationlength_model(correlationlength_model_)
{
    // dimension
    int dim = lattice->dim();
    if (not(dim == 2))
    {
        std::cout << "SquaredShiftedLaplaceFDOperator only implemented for d=2" << std::endl;
        exit(-1);
    }
    // number of matrix rows
    int nrow = lattice->Nvertex;
    // shape of lattice
    Eigen::VectorXi shape = lattice->shape();
    // grid spacings in all directions
    Eigen::VectorXd h(dim);
    // inverse squared grid spacings in all dimensions (required in 2nd order term)
    Eigen::VectorXd hinv2(dim);
    // cell volume
    double cell_volume = 1.0;
    for (int d = 0; d < dim; ++d)
    {
        h[d] = 1. / double(shape[d]);
        hinv2[d] = 1. / (h[d] * h[d]);
        cell_volume *= h[d];
    }
    typedef Eigen::Triplet<double> T;
    std::vector<T> tripletList;
    tripletList.reserve(13 * nrow);
    /* stencil elements for Laplacian */
    double stencil_laplacian[2][2];
    stencil_laplacian[0][0] = -2 * (hinv2[0] + hinv2[1]);
    stencil_laplacian[1][0] = hinv2[0];
    stencil_laplacian[0][1] = hinv2[1];
    /* stencil elements for Laplacian^2 */
    double stencil_squared_laplacian[3][3];
    stencil_squared_laplacian[0][0] = 6 * (hinv2[0] * hinv2[0] + hinv2[1] * hinv2[1]) + 8 * hinv2[0] * hinv2[1];
    stencil_squared_laplacian[1][0] = -4 * hinv2[0] * (hinv2[0] + hinv2[1]);
    stencil_squared_laplacian[0][1] = -4 * hinv2[1] * (hinv2[0] + hinv2[1]);
    stencil_squared_laplacian[2][0] = hinv2[0] * hinv2[0];
    stencil_squared_laplacian[0][2] = hinv2[1] * hinv2[1];
    stencil_squared_laplacian[1][1] = 2 * hinv2[0] * hinv2[1];
    for (unsigned int ell = 0; ell < nrow; ++ell)
    {
        Eigen::VectorXd x = lattice->vertex_coordinates(ell);
        double alpha_b = correlationlength_model->kappa_invsq(x);
        double diagonal = (alpha_b * alpha_b - 2. * alpha_b * stencil_laplacian[0][0] + stencil_squared_laplacian[0][0]) * cell_volume;
        /* Loop over 5x5 stencil and only treat entries in this diamond:
         *
         *          . . x . .
         *          . x x x .
         *          x x . x x
         *          . x x x .
         *          . . x . .
         */
        for (int j = -2; j <= 2; ++j)
        {
            for (int k = -2; k <= 2; ++k)
            {
                if ((abs(j) + abs(k) > 2) or ((j == 0) and (k == 0)))
                    continue;
                Eigen::Vector2i shift;
                shift[0] = j;
                shift[1] = k;
                unsigned int ell_shifted;
                if (lattice->shifted_vertex_is_internal_vertex(ell, shift, ell_shifted))
                {
                    double local_matrix_element = stencil_squared_laplacian[abs(j)][abs(k)];
                    if (abs(j) + abs(k) == 1)
                        local_matrix_element += -2. * alpha_b * stencil_laplacian[abs(j)][abs(k)];
                    tripletList.push_back(T(ell, ell_shifted, local_matrix_element * cell_volume));
                }
                else if (abs(j) + abs(k) == 1)
                {
                    /* deal with homogeneous Neumann BCs: is one of the stencil entries
                     * (+1,0), (-1,0), (0,+1), (0,-1) touches the boundary, the value for
                     * the entry (+2,0), (-2,0), (0,+2), (0,-2) needs to be added to the
                     * diagonal.
                     */
                    diagonal += stencil_squared_laplacian[2 * abs(j)][2 * abs(k)] * cell_volume;
                }
            }
        }
        tripletList.push_back(T(ell, ell, diagonal));
    }
    A_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
}