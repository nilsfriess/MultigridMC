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

inline double g(double z)
{

    return 2500 * z * z * z * z * (1 - z) * (1 - z) * exp(-8 * z);
}

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
    std::shared_ptr<SmootherFactory> presmoother_factory = std::make_shared<SORSmootherFactory>(smoother_params.omega,
                                                                                                forward);
    std::shared_ptr<SmootherFactory> postsmoother_factory = std::make_shared<SORSmootherFactory>(smoother_params.omega,
                                                                                                 backward);
    std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory = std::make_shared<IntergridOperatorLinearFactory>();
    std::shared_ptr<LinearSolverFactory> coarse_solver_factory = std::make_shared<CholeskySolverFactory>();
    std::shared_ptr<Preconditioner> multigrid_preconditioner = std::make_shared<MultigridPreconditioner>(linear_operator,
                                                                                                         multigrid_params,
                                                                                                         presmoother_factory,
                                                                                                         postsmoother_factory,
                                                                                                         intergrid_operator_factory,
                                                                                                         coarse_solver_factory);
    std::cout << std::endl;
    // Run sampling experiments
    LoopSolver solver(linear_operator,
                      multigrid_preconditioner,
                      iterative_solver_params);
    // Create states
    unsigned int ndof = linear_operator->get_ndof();
    Eigen::VectorXd x_exact(ndof);
    Eigen::VectorXd x(ndof);
    unsigned int seed = 1482817;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    double volume = lattice->cell_volume();
    Eigen::VectorXi shape = lattice->shape();
    Eigen::VectorXd h_lat(2);
    h_lat[0] = 1 / double(shape[0]);
    h_lat[1] = 1 / double(shape[1]);
    for (unsigned int ell = 0; ell < lattice->Nvertex; ++ell)
    {
        Eigen::VectorXi coord = lattice->vertexidx_linear2euclidean(ell);
        Eigen::VectorXd x = h_lat.cwiseProduct(coord.cast<double>());
        x_exact[ell] = g(x[0]) * g(x[1]);
    }

    Eigen::VectorXd b(ndof);
    linear_operator->apply(x_exact, b);
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
    vtk_writer->add_state(x_exact, "exact");
    vtk_writer->add_state(x, "numerical");
    vtk_writer->add_state(x - x_exact, "error");
    vtk_writer->write();
}
