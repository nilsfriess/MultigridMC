#include "measured_operator.hh"

/** @file measured_operator.cc
 *
 * @brief Implementation of measured_operator.hh
 */

/* Create a new instance */
MeasuredOperator::MeasuredOperator(const std::shared_ptr<LinearOperator> base_operator_,
                                   const std::vector<Eigen::VectorXd> measurement_locations_,
                                   const Eigen::MatrixXd Sigma_,
                                   const bool ignore_measurement_cross_correlations_,
                                   const bool measure_average_,
                                   const double sigma_average_) : LinearOperator(base_operator_->get_lattice(),
                                                                                 measurement_locations_.size() + measure_average_),
                                                                  base_operator(base_operator_)
{
    A_sparse = base_operator->get_sparse();
    unsigned int nrow = base_operator->get_lattice()->Nvertex;
    unsigned int n_measurements = measurement_locations_.size();
    Sigma_inv = Eigen::MatrixXd(n_measurements + measure_average_, n_measurements + measure_average_);
    Sigma_inv.setZero();
    DenseMatrixType Sigma_local;
    if (ignore_measurement_cross_correlations_)
    {
        Sigma_local = Sigma_.diagonal().asDiagonal();
    }
    else
    {
        Sigma_local = Sigma_;
    }
    Sigma_inv(Eigen::seqN(0, n_measurements), Eigen::seqN(0, n_measurements)) = Sigma_local.inverse();
    typedef Eigen::Triplet<double> T;
    std::vector<T> triplet_list;
    // <<<<<<<<<<<<<<<<<<<<< READ THIS FROM DISK
    double radius = 0.1;
    // <<<<<<<<<<<<<<<<<<<<< READ THIS FROM DISK
    for (int k = 0; k < n_measurements; ++k)
    {
        Eigen::VectorXd x_0 = measurement_locations_[k];
        Eigen::SparseVector<double> r_meas = measurement_vector(x_0, radius);
        for (Eigen::SparseVector<double>::InnerIterator it(r_meas); it; ++it)
            triplet_list.push_back(T(it.col(), k, it.value()));
    }
    if (measure_average_)
    {
        for (int j = 0; j < nrow; ++j)
        {
            triplet_list.push_back(T(j, n_measurements, 1. / nrow));
        }
        Sigma_inv(n_measurements, n_measurements) = 1. / sigma_average_;
    }
    B.setFromTriplets(triplet_list.begin(), triplet_list.end());
    Sigma_inv_BT = Sigma_inv.sparseView() * B.transpose();
}

/* Create measurement vector in dual space */
Eigen::SparseVector<double> MeasuredOperator::measurement_vector(const Eigen::VectorXd x0, const double radius) const
{
    Eigen::SparseVector<double> r_meas(lattice->Nvertex);
    // dimension
    int dim = lattice->dim();
    // shape of lattice
    Eigen::VectorXi shape = lattice->shape();
    // grid spacings in all directions
    Eigen::VectorXd h(dim);
    // cell volume
    double cell_volume = 1.0;
    for (int d = 0; d < dim; ++d)
    {
        h[d] = 1. / double(shape[d]);
        cell_volume *= h[d];
    }
    GaussLegendreQuadrature quadrature(dim, 1);
    std::vector<double> quad_weights = quadrature.get_weights();
    std::vector<Eigen::VectorXd> quad_points = quadrature.get_points();
    std::vector<int> basis_idx_1d{0, 1}; // indices used to identify the basis functions
    // Vector of all possible basis indices in d dimensions
    std::vector<std::vector<int>> basis_idx = cartesian_product(basis_idx_1d, dim);

    for (int cell_idx = 0; cell_idx < lattice->Ncell; ++cell_idx)
    { // loop over all cells of the lattice
        Eigen::VectorXi cell_coord = lattice->cellidx_linear2euclidean(cell_idx);
        bool overlap = false;
        for (auto it = basis_idx.begin(); it != basis_idx.end(); ++it)
        { // loop over all corners of the cell and check whether one of them overlaps with
          // a ball of radius R around the point x_0
            Eigen::Map<Eigen::VectorXi> omega(it->data(), dim);
            Eigen::VectorXd x_corner = h.cwiseProduct((cell_coord + omega).cast<double>());
            if ((x_corner - x0).norm() < radius)
            {
                overlap = true;
                break;
            }
        }
        if (not overlap) // move on to next cell if there is no overlap
            continue;

        for (auto it = basis_idx.begin(); it != basis_idx.end(); ++it)
        { // now loop over the basis functions associated with the corners
          // of the cell
            Eigen::Map<Eigen::VectorXi> alpha(it->data(), dim);
            unsigned int ell;
            if (lattice->corner_is_internal_vertex(cell_idx, alpha, ell))
            { // found an interior vertex
                double local_entry = 0.0;
                for (int j = 0; j < quad_points.size(); ++j)
                { // Loop over all quadrature points
                    Eigen::VectorXd xhat = quad_points[j];
                    // Convert integer-valued coordinates to coordinates in domain
                    Eigen::VectorXd x = h.cwiseProduct(xhat + cell_coord.cast<double>());
                    // evaluate basis function
                    double xi = (x - x0).norm() / radius;
                    if (xi < 1.0)
                    {
                        double phihat = f_meas(xi);
                        for (int j = 0; j < dim; ++j)
                        {
                            phihat *= (alpha[j] == 0) ? (1.0 - xhat[j]) : xhat[j];
                        }
                        local_entry += phihat * quad_weights[j] * cell_volume;
                    }
                }
                r_meas.coeffRef(ell) = local_entry;
            }
        }
    }
    return r_meas;
}