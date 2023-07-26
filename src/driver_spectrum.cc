#include <iostream>
#include <algorithm>
#include <vector>
#include <fstream>
#include <Eigen/Eigenvalues>

#include "config.h"
#include "lattice/lattice2d.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/diffusion_operator.hh"
#include "linear_operator/measured_operator.hh"
#include "auxilliary/parameters.hh"

/* *********************************************************************** *
 *                                M A I N
 * *********************************************************************** */
int main(int argc, char *argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: " << argv[0] << " CONFIGURATIONFILE" << std::endl;
        exit(-1);
    }
    std::string filename(argv[1]);
    std::cout << "Reading parameters from file \'" << filename << "\'" << std::endl;
    LatticeParameters lattice_params;
    DiffusionParameters diffusion2d_params;
    MeasurementParameters measurement_params;
    lattice_params.read_from_file(filename);
    diffusion2d_params.read_from_file(filename);
    measurement_params.read_from_file(filename);

    // Construct lattice and linear operator
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(lattice_params.nx,
                                                                     lattice_params.ny);
    std::shared_ptr<DiffusionOperator2d> diffusion_operator = std::make_shared<DiffusionOperator2d>(lattice,
                                                                                                    diffusion2d_params.alpha_K,
                                                                                                    diffusion2d_params.beta_K,
                                                                                                    diffusion2d_params.alpha_b,
                                                                                                    diffusion2d_params.beta_b);
    std::shared_ptr<MeasuredOperator> linear_operator = std::make_shared<MeasuredOperator>(diffusion_operator,
                                                                                           measurement_params.measurement_locations,
                                                                                           measurement_params.covariance,
                                                                                           measurement_params.ignore_measurement_cross_correlations,
                                                                                           measurement_params.measure_global,
                                                                                           measurement_params.sigma_global);
    LinearOperator::DenseMatrixType covariance = linear_operator->covariance();
    typedef Eigen::EigenSolver<LinearOperator::DenseMatrixType> EigenSolver;
    EigenSolver eigen_solver(covariance, false);
    EigenSolver::EigenvalueType eigen_values = eigen_solver.eigenvalues();
    unsigned int n = eigen_values.rows();
    std::vector<double> v(n);
    for (int j = 0; j < n; ++j)
    {
        v[j] = eigen_values[j].real();
    }
    std::sort(v.begin(), v.end());
    std::ofstream outfile;
    outfile.open("spectrum.csv");
    for (int j = 0; j < n; ++j)
    {
        outfile << v[j];
        if (j < n - 1)
        {
            outfile << ", ";
        }
        else
        {
            outfile << std::endl;
        }
    }
    outfile.close();
}
