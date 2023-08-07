#include "shiftedlaplace_fem_operator.hh"

/** @file shiftedlaplace_fem_operator.cc
 *
 * @brief Implementation of shiftedlaplace_fem_operator.hh
 */

/*  Create a new instance */
DiffusionOperator::DiffusionOperator(const std::shared_ptr<Lattice> lattice_,
                                     const double alpha_K_,
                                     const double beta_K_,
                                     const double alpha_b_,
                                     const double beta_b_,
                                     const int verbose) : LinearOperator(lattice_),
                                                          alpha_K(alpha_K_),
                                                          beta_K(beta_K_),
                                                          alpha_b(alpha_b_),
                                                          beta_b(beta_b_)
{
    typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePointType;
    TimePointType t_start;
    if (verbose > 0)
    {
        t_start = std::chrono::high_resolution_clock::now();
    }
    // dimension
    int dim = lattice->dim();
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
    // offsets (=shifts) in one dimension
    std::vector<int> shift_1d{-1, 0, +1};
    // Vector of all possible d-dimensional offsets (=shifts) from a given lattice point
    std::vector<std::vector<int>> shifts = cartesian_product(shift_1d, dim);

    /* ==== STEP 1 ==== Initialise sparsity structure of matrix
     *
     * Initialise all potentially non-zero entries of the matrix to zero; these
     * entries will be populated with concrete values during matrix assembly.
     */

    typedef Eigen::Triplet<double> T;
    std::vector<T> triplet_list;
    triplet_list.reserve(nrow * int(pow(3, dim))); // there are (up to) 3^d entries per matrix row
    for (unsigned int ell_row = 0; ell_row < lattice->Nvertex; ++ell_row)
    { // loop over all internal vertices
        for (auto it = shifts.begin(); it != shifts.end(); ++it)
        { // loop over all possible shifts
            Eigen::Map<Eigen::VectorXi> shift(it->data(), dim);
            Eigen::VectorXi p = lattice->vertexidx_linear2euclidean(ell_row) + shift;
            // Check whether the shift takes us to a point that is no longer an interior vertex
            bool valid_shift = true;
            for (int d = 0; d < dim; ++d)
                valid_shift = valid_shift && (not((p[d] == 0) or (p[d] == shape[d])));
            if (valid_shift)
            {
                unsigned int ell_col = lattice->vertexidx_euclidean2linear(p);
                triplet_list.push_back(T(ell_row, ell_col, 0.0));
            }
        }
    }
    A_sparse.setFromTriplets(triplet_list.begin(), triplet_list.end());

    /* ==== STEP 2 ==== global matrix assembly
     *
     * Loop over all cells and all pairs (alpha,beta) of basis functions inside a cell;
     */
    GaussLegendreQuadrature quadrature(dim, 1);
    std::vector<double> quad_weights = quadrature.get_weights();
    std::vector<Eigen::VectorXd> quad_points = quadrature.get_points();
    std::vector<int> basis_idx_1d{0, 1}; // indices used to identify the basis functions
    // Vector of all possible basis indices in d dimensions
    std::vector<std::vector<int>> basis_idx = cartesian_product(basis_idx_1d, dim);
    std::vector<double> phi_phi;
    std::vector<double> gradphi_gradphi;
    for (auto it_alpha = basis_idx.begin(); it_alpha != basis_idx.end(); ++it_alpha)
    { // Loop over all basis functions
        Eigen::Map<Eigen::VectorXi> alpha(it_alpha->data(), dim);
        for (auto it_beta = basis_idx.begin(); it_beta != basis_idx.end(); ++it_beta)
        {
            Eigen::Map<Eigen::VectorXi> beta(it_beta->data(), dim);
            for (int j = 0; j < quad_points.size(); ++j)
            { // Loop over all quadrature points
                Eigen::VectorXd xhat = quad_points[j];
                phi_phi.push_back(phi(alpha, xhat) * phi(beta, xhat));
                gradphi_gradphi.push_back(grad_phi(alpha, xhat).dot(hinv2.cwiseProduct(grad_phi(beta, xhat))));
            }
        }
    }
    // Now loop over all cells and assemble the matrix
    for (unsigned int cell_idx = 0; cell_idx < lattice->Ncell; ++cell_idx)
    { // loop over all cells
        unsigned int count = 0;
        Eigen::VectorXi cell_coord = lattice->cellidx_linear2euclidean(cell_idx);
        for (auto it_alpha = basis_idx.begin(); it_alpha != basis_idx.end(); ++it_alpha)
        {
            Eigen::Map<Eigen::VectorXi> alpha(it_alpha->data(), dim);
            for (auto it_beta = basis_idx.begin(); it_beta != basis_idx.end(); ++it_beta)
            {
                Eigen::Map<Eigen::VectorXi> beta(it_beta->data(), dim);
                // Check whether the matrix entry is valid, i.e. whether it couples unknowns
                // associated with interior vertices
                unsigned int ell_row, ell_col;
                if (lattice->corner_is_internal_vertex(cell_idx, alpha, ell_row) and
                    lattice->corner_is_internal_vertex(cell_idx, beta, ell_col))
                {
                    double local_matrix_entry = 0.0;
                    for (int j = 0; j < quad_points.size(); ++j)
                    { // Loop over all quadrature points
                        Eigen::VectorXd xhat = quad_points[j];
                        // Convert integer-valued coordinates to coordinates in domain
                        Eigen::VectorXd x = h.cwiseProduct(xhat + cell_coord.cast<double>());

                        /* increment local matrix entry by
                         *
                         *    ( b(x_q) * phi_alpha(xhat_q) * phi_beta(xhat_q)
                         *    + K(x_q) * sum_{j=0}^d h_j^{-2} * dphi_alpha(xhat_q)/dxhat_j * dphi_beta(xhat_q)/dxhat_j )
                         *        * h_0 * h_1 * ... * h_{d-1} * w_q
                         */
                        local_matrix_entry += (b_zero(x) * phi_phi[count] +
                                               K_diff(x) * gradphi_gradphi[count]) *
                                              quad_weights[j];
                        count++;
                    }
                    A_sparse.coeffRef(ell_row, ell_col) += local_matrix_entry * cell_volume;
                }
                else
                {
                    count += quad_points.size();
                }
            }
        }
    }
    if (verbose > 0)
    {
        TimePointType t_finish = std::chrono::high_resolution_clock::now();
        double t_elapsed = 1.E-3 * std::chrono::duration_cast<std::chrono::milliseconds>(t_finish - t_start).count();
        int buffersize = 64;
        char buffer[buffersize];
        snprintf(buffer, buffersize, "diffusion operator assembly time = %8.2f s", t_elapsed);
        std::cout << std::endl
                  << buffer << std::endl
                  << std::endl;
    }
}

/** @brief Diffusion coefficient */
double DiffusionOperator::K_diff(const Eigen::VectorXd x) const
{
    if (abs(beta_K / alpha_K) < 1.E-12)
        return alpha_K;
    double value = beta_K;
    for (int j = 0; j < x.size(); ++j)
        value *= sin(2 * M_PI * x[j]);
    value += alpha_K;
    return value;
}

/** @brief Zero order term */
double DiffusionOperator::b_zero(const Eigen::VectorXd x) const
{
    if (abs(beta_b / alpha_b) < 1.E-12)
        return alpha_b;
    double value = beta_b;
    for (int j = 0; j < x.size(); ++j)
        value *= cos(2 * M_PI * x[j]);
    value += alpha_b;
    return value;
}

/* Evaluate basis function in reference cell */
double DiffusionOperator::phi(Eigen::VectorXi alpha, Eigen::VectorXd xhat) const
{
    int dim = alpha.size();
    double phihat = 1.0;
    for (int j = 0; j < dim; ++j)
    {
        phihat *= (alpha[j] == 0) ? (1.0 - xhat[j]) : xhat[j];
    }
    return phihat;
}

/* Evaluate gradients of basis function in reference cell */
Eigen::VectorXd DiffusionOperator::grad_phi(Eigen::VectorXi alpha, Eigen::VectorXd xhat) const
{
    int dim = alpha.size();
    Eigen::VectorXd grad_phihat(dim);
    for (int k = 0; k < dim; ++k)
    {
        double grad_phihat_k = 1.0;
        for (int j = 0; j < alpha.size(); ++j)
        {
            if (j == k)
            {
                grad_phihat_k *= (alpha[j] == 0) ? -1.0 : +1.0;
            }
            else
            {
                grad_phihat_k *= (alpha[j] == 0) ? (1.0 - xhat[j]) : xhat[j];
            }
        }
        grad_phihat[k] = grad_phihat_k;
    }
    return grad_phihat;
}