#include <iostream>
#include <random>
#include <cmath>
#include <Eigen/Dense>

#include "config.h"
#include "lattice/lattice2d.hh"
#include "lattice/lattice3d.hh"
#include "smoother/ssor_smoother.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/correlationlength_model.hh"
#include "linear_operator/shiftedlaplace_fem_operator.hh"
#include "linear_operator/shiftedlaplace_fd_operator.hh"
#include "linear_operator/squared_shiftedlaplace_fd_operator.hh"
#include "linear_operator/measured_operator.hh"
#include "intergrid/intergrid_operator_linear.hh"
#include "solver/linear_solver.hh"
#include "solver/loop_solver.hh"
#include "solver/cholesky_solver.hh"
#include "preconditioner/multigrid_preconditioner.hh"
#include "auxilliary/parameters.hh"
#include "auxilliary/vtk_writer2d.hh"
#include "auxilliary/vtk_writer3d.hh"

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
    std::cout << std::endl;
    std::cout << "+------------------+" << std::endl;
    std::cout << "! Multigrid solver !" << std::endl;
    std::cout << "+------------------+" << std::endl;
    std::cout << std::endl;
    std::string filename(argv[1]);
    std::cout << "Reading parameters from file \'" << filename << "\'" << std::endl;
    GeneralParameters general_params;
    LatticeParameters lattice_params;
    SmootherParameters smoother_params;
    IterativeSolverParameters iterative_solver_params;
    MultigridParameters multigrid_params;
    PriorParameters prior_params;
    ConstantCorrelationLengthModelParameters constantcorrelationlengthmodel_params;
    PeriodicCorrelationLengthModelParameters periodiccorrelationlengthmodel_params;
    MeasurementParameters measurement_params;
    general_params.read_from_file(filename);
    lattice_params.read_from_file(filename);
    smoother_params.read_from_file(filename);
    multigrid_params.read_from_file(filename);
    iterative_solver_params.read_from_file(filename);
    prior_params.read_from_file(filename);
    constantcorrelationlengthmodel_params.read_from_file(filename);
    periodiccorrelationlengthmodel_params.read_from_file(filename);
    measurement_params.read_from_file(filename);

    if (measurement_params.dim != general_params.dim)
    {
        std::cout << "ERROR: dimension of measurement locations differs from problem dimension" << std::endl;
        exit(-1);
    }
#if (defined EIGEN_USE_BLAS && defined EIGEN_USE_LAPACKE)
    std::cout << "Compiled with BLAS/LAPACK support for Eigen." << std::endl
              << std::endl;
#else  // EIGEN_USE_BLAS && EIGEN_USE_LAPACKE
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << "WARNING: Compiled without BLAS/LAPACK support for Eigen." << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << std::endl;
#endif // EIGEN_USE_BLAS && EIGEN_USE_LAPACKE

#ifndef NCHOLMOD
    std::cout << "Using sparse Cholesky factorisation from CholMod." << std::endl;
#else  // NCHOLMOD
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << "WARNING: Falling back on Eigen's SimplicalLLT Cholesky factorisation." << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << std::endl;
#endif // NCHOLMOD

    // Construct lattice and linear operator
    std::shared_ptr<Lattice> lattice;
    if (general_params.dim == 2)
    {
        lattice = std::make_shared<Lattice2d>(lattice_params.nx,
                                              lattice_params.ny);
    }
    else if (general_params.dim == 3)
    {
        lattice = std::make_shared<Lattice3d>(lattice_params.nx,
                                              lattice_params.ny,
                                              lattice_params.nz);
    }
    else
    {
        std::cout << "ERROR: Invalid dimension : " << general_params.dim << std::endl;
        exit(-1);
    }
    std::shared_ptr<CorrelationLengthModel> correlationlengthmodel;
    if (prior_params.correlationlength_model == "constant")
    {
        correlationlengthmodel = std::make_shared<ConstantCorrelationLengthModel>(constantcorrelationlengthmodel_params);
    }
    else if (prior_params.correlationlength_model == "periodic")
    {
        correlationlengthmodel = std::make_shared<PeriodicCorrelationLengthModel>(periodiccorrelationlengthmodel_params);
    }
    else
    {
        std::cout << "Error: invalid correlationlengthmodel \'" << prior_params.correlationlength_model << "\'" << std::endl;
        exit(-1);
    }
    std::shared_ptr<LinearOperator> prior_operator;
    if (prior_params.pde_model == "shiftedlaplace_fem")
    {
        prior_operator = std::make_shared<ShiftedLaplaceFEMOperator>(lattice, correlationlengthmodel, 1);
    }
    else if (prior_params.pde_model == "shiftedlaplace_fd")
    {
        prior_operator = std::make_shared<ShiftedLaplaceFDOperator>(lattice, correlationlengthmodel, 1);
    }
    else if (prior_params.pde_model == "squared_shiftedlaplace_fd")
    {
        prior_operator = std::make_shared<SquaredShiftedLaplaceFDOperator>(lattice, correlationlengthmodel, 1);
    }
    else
    {
        std::cout << "Error: invalid prior \'" << prior_params.pde_model << "\'" << std::endl;
        exit(-1);
    }

    std::shared_ptr<MeasuredOperator> posterior_operator = std::make_shared<MeasuredOperator>(prior_operator,
                                                                                              measurement_params);
    std::shared_ptr<LinearOperator> linear_operator;
    if (general_params.operator_name == "prior")
    {
        linear_operator = prior_operator;
    }
    else if (general_params.operator_name == "posterior")
    {
        linear_operator = posterior_operator;
    }
    else
    {
        std::cout << "ERROR: invalid operator : " << general_params.operator_name << std::endl;
        exit(-1);
    }
    //   Construct smoothers
    /* prepare measurements */
    std::shared_ptr<Preconditioner> multigrid_preconditioner = std::make_shared<MultigridPreconditioner>(linear_operator,
                                                                                                         multigrid_params);
    std::cout << std::endl;
    // Run sampling experiments
    LoopSolver solver(linear_operator,
                      multigrid_preconditioner,
                      iterative_solver_params);
    // Create states
    unsigned int ndof = linear_operator->get_ndof();
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd b(ndof);
    unsigned int seed = 1482817;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    int dim = lattice->dim();
    for (unsigned int ell = 0; ell < lattice->Nvertex; ++ell)
    {
        b[ell] = normal_dist(rng);
    }
    solver.apply(b, x);
    std::shared_ptr<VTKWriter> vtk_writer;
    if (general_params.dim == 2)
    {
        vtk_writer = std::make_shared<VTKWriter2d>("solution.vtk", lattice, 1);
    }
    else
    {
        vtk_writer = std::make_shared<VTKWriter3d>("solution.vtk", lattice, 1);
    }
    vtk_writer->add_state(x, "numerical");
    vtk_writer->write();
}
