#include "shiftedbiharmonic_operator.hh"

/** @file shiftedbiharmonic_operator.cc
 *
 * @brief Implementation of shiftedbiharmonic_operator.hh
 */

/*  Create a new instance */
ShiftedBiharmonicOperator::ShiftedBiharmonicOperator(const std::shared_ptr<Lattice> lattice_,
                                                     const double alpha_K_,
                                                     const double alpha_b_,
                                                     const int verbose) : LinearOperator(lattice_),
                                                                          alpha_K(alpha_K_),
                                                                          alpha_b(alpha_b_)
{
    // dimension
    int dim = lattice->dim();
    if (not(dim == 2))
    {
        std::cout << "ShiftedBiharmonicOperator only implemented for d=2" << std::endl;
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
    double cell_volume = lattice->cell_volume();
    for (int d = 0; d < dim; ++d)
    {
        h[d] = 1. / double(shape[d]);
        hinv2[d] = 1. / (h[d] * h[d]);
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
        /* Loop over 5x5 stencil and only treat entries in this diamond:
         *
         *          . . x . .
         *          . x x x .
         *          x x . x x
         *          . x x x .
         *          . . x . .
         */
        std::map<std::pair<unsigned int, unsigned int>, double> m;
        m[std::make_pair<int, int>(ell, ell)] = (alpha_b * alpha_b - 2 * alpha_K * alpha_b * stencil_laplacian[0][0] + alpha_K * alpha_K * stencil_squared_laplacian[0][0]) * cell_volume;
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
                    double local_matrix_element = alpha_K * alpha_K * stencil_squared_laplacian[abs(j)][abs(k)];
                    if (abs(j) + abs(k) == 1)
                        local_matrix_element += -2 * alpha_K * alpha_b * stencil_laplacian[abs(j)][abs(k)];
                    m[std::make_pair<int, int>(ell, ell_shifted)] = local_matrix_element * cell_volume;
                }
            }
        }
        for (int j = -1; j <= 1; ++j)
        {
            for (int k = -1; k <= 1; ++k)
            {
                if (abs(j) + abs(k) == 1)
                {
                    Eigen::Vector2i shift;
                    shift[0] = j;
                    shift[1] = k;
                    unsigned int ell_shifted;
                    if (not(lattice->shifted_vertex_is_internal_vertex(ell, shift, ell_shifted)))
                    {
                        double neg_value = alpha_K * alpha_K * stencil_squared_laplacian[2 * abs(j)][2 * abs(k)] * cell_volume;
                        unsigned int ell_neg_shifted;
                        m[std::make_pair<int, int>(ell, ell)] += 6 * neg_value;
                        Eigen::Vector2i neg_shift;
                        neg_shift[0] = -j;
                        neg_shift[1] = -k;
                        ell_neg_shifted = lattice->shift_vertexidx(ell, neg_shift);
                        m[std::make_pair<int, int>(ell, ell_neg_shifted)] -= 2 * neg_value;
                        neg_shift[0] = -2 * j;
                        neg_shift[1] = -2 * k;
                        ell_neg_shifted = lattice->shift_vertexidx(ell, neg_shift);
                        m[std::make_pair<int, int>(ell, ell_neg_shifted)] += 1. / 3. * neg_value;
                    }
                }
            }
        }
        for (auto it = m.begin(); it != m.end(); ++it)
        {
            tripletList.push_back(T(it->first.first, it->first.second, it->second));
        }
    }
    A_sparse.setFromTriplets(tripletList.begin(), tripletList.end());
}