#ifndef MEASURED_OPERATOR_HH
#define MEASURED_OPERATOR_HH MEASURED_OPERATOR_HH

#include <vector>
#include <Eigen/Dense>
#include "linear_operator.hh"
#include "lattice/lattice.hh"

/** @file measured_operator.hh
 *
 * @brief Contains class for measured operator
 */

/** @brief linear operator with measurements
 *
 * Assume that we measured data as Y = B^T X + E, where X is drawn from a prior
 * distribution N(xbar,Q^{-1}) and E is draw from an (independent) multivariate normal
 * distribution N(0,Sigma) with covariance Sigma. The conditional distribution of X given y
 * is then a multivariate normal distribution with mean
 *
 *   x_{X|y} = xbar + Q^{-1} B (Sigma + B^T Q^{-1} B)^{-1} (y - B^T xbar)
 *
 * and precision matrix
 *
 *   Q_{X|y} = Q + B Sigma^{-1} B^T.
 *
 */
class MeasuredOperator : public LinearOperator
{
public:
    /** @brief Create a new instance
     *
     * Populates matrix entries across the grid
     *
     * @param[in] base_operator_ underlying linear operator
     * @param[in] measurement_locations_ coordinates of locations where the field is measured
     * @param[in] Sigma_ covariance matrix of measurements
     * @param[in] ignore_measurement_cross_correlations_ ignore all off-diagonal entries in the
     *            covariance matrix Sigma
     * @param[in] measure_average_ measure the average across the entire domain
     * @param[in] sigma_average_ variance of global average measurement
     */
    MeasuredOperator(const std::shared_ptr<LinearOperator> base_operator_,
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
        for (int k = 0; k < n_measurements; ++k)
        {
            Eigen::VectorXd x_loc = measurement_locations_[k];
            Eigen::VectorXi x_loc_int(x_loc.size());
            Eigen::VectorXi shape = lattice->shape();
            for (int j = 0; j < x_loc.size(); ++j)
            {
                x_loc_int[j] = int(round(x_loc[j] * shape[j]));
            }
            unsigned int ell = lattice->cellidx_euclidean2linear(x_loc_int);
            triplet_list.push_back(T(ell, k, 1.0));
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

    /** @brief Compute posterior mean
     *
     * @param[in] xbar prior mean
     * @param[in] y measured values
     */
    Eigen::VectorXd posterior_mean(const Eigen::VectorXd &xbar,
                                   const Eigen::VectorXd &y)
    {
        Eigen::SimplicialLLT<SparseMatrixType> solver;
        solver.compute(A_sparse);
        // Compute Bbar = Q^{-1} B
        Eigen::MatrixXd Bbar = solver.solve(B);
        Eigen::VectorXd x_post = xbar + Bbar * (Sigma_inv.inverse() + B.transpose() * Bbar).inverse() * (y - B.transpose() * xbar);
        return x_post;
    }

protected:
    /** @brief underlying linear operator */
    std::shared_ptr<LinearOperator> base_operator;
};

#endif // MEASURED_OPERATOR_HH