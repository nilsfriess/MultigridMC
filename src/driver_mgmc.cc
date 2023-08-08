#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <chrono>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/QR>

#include "config.h"
#include "lattice/lattice2d.hh"
#include "lattice/lattice3d.hh"
#include "smoother/ssor_smoother.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/shiftedlaplace_fem_operator.hh"
#include "linear_operator/shiftedlaplace_fd_operator.hh"
#include "linear_operator/squared_shiftedlaplace_fd_operator.hh"
#include "linear_operator/measured_operator.hh"
#include "intergrid/intergrid_operator_linear.hh"
#include "sampler/cholesky_sampler.hh"
#include "sampler/ssor_sampler.hh"
#include "sampler/multigridmc_sampler.hh"
#include "solver/loop_solver.hh"
#include "preconditioner/multigrid_preconditioner.hh"
#include "auxilliary/parameters.hh"
#include "auxilliary/vtk_writer2d.hh"
#include "auxilliary/vtk_writer3d.hh"
#include "auxilliary/statistics.hh"

/** @brief generate a number of samples, meaure runtime and write timeseries to disk
 *
 * @param[in] sampler sampler to be used
 * @param[in] sampling_params parameters for sampling
 * @param[in] measurement_params parameters for measurements
 * @param[in] filename name of file to write to
 */
void measure_sampling_time(std::shared_ptr<Sampler> sampler,
                           const SamplingParameters &sampling_params,
                           const MeasurementParameters &measurement_params,
                           const std::string filename)
{
    const std::shared_ptr<MeasuredOperator> linear_operator = std::dynamic_pointer_cast<MeasuredOperator>(sampler->get_linear_operator());
    unsigned int ndof = linear_operator->get_ndof();
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    Eigen::VectorXd xbar(ndof);
    xbar.setZero();
    Eigen::VectorXd y(measurement_params.n + measurement_params.measure_global);
    y(Eigen::seqN(0, measurement_params.n)) = measurement_params.mean;
    if (measurement_params.measure_global)
        y(measurement_params.n) = measurement_params.mean_global;
    Eigen::VectorXd x_post = linear_operator->posterior_mean(xbar, y);
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    x.setZero();
    linear_operator->apply(x_post, f);
    for (int k = 0; k < sampling_params.nwarmup; ++k)
    {
        sampler->apply(f, x);
    };
    std::vector<double> data(sampling_params.nsamples);

    Eigen::SparseVector<double> sample_vector = linear_operator->measurement_vector(measurement_params.sample_location,
                                                                                    measurement_params.radius);

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < sampling_params.nsamples; ++k)
    {
        sampler->apply(f, x);
        data[k] = sample_vector.dot(x);
    }
    auto t_finish = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_finish - t_start).count() / (1.0 * sampling_params.nsamples);
    std::cout << "  time per sample = " << t_elapsed << " ms" << std::endl;
    std::ofstream out;
    out.open(filename);
    for (auto it = data.begin(); it != data.end(); ++it)
        out << *it << std::endl;
    // Compute mean and variance of measurements
    double x_avg = 0.0;
    double xsq_avg = 0.0;
    for (int k = 0; k < sampling_params.nsamples; ++k)
    {
        x_avg += (data[k] - x_avg) / (k + 1.0);
        xsq_avg += (data[k] * data[k] - xsq_avg) / (k + 1.0);
    }
    double variance = xsq_avg - x_avg * x_avg;
    double x_error = sqrt(variance / sampling_params.nsamples);
    printf("  mean     = %12.4e +/- %12.4e [ignoring autocorrelations]\n", x_avg, x_error);
    printf("  variance = %12.4e\n\n", variance);

    out.close();
}

/** @brief compute mean and variance field
 *
 * @param[in] sampler sampler to be used
 * @param[in] sampling_params parameters for sampling
 * @param[in] measurement_params parameters for measurements
 */
void posterior_statistics(std::shared_ptr<Sampler> sampler,
                          const SamplingParameters &sampling_params,
                          const MeasurementParameters &measurement_params)
{
    const std::shared_ptr<MeasuredOperator> linear_operator = std::dynamic_pointer_cast<MeasuredOperator>(sampler->get_linear_operator());
    unsigned int ndof = linear_operator->get_ndof();

    // prior mean (set to zero)
    Eigen::VectorXd xbar(ndof);
    xbar.setZero();
    Eigen::VectorXd y(measurement_params.n + measurement_params.measure_global);
    y(Eigen::seqN(0, measurement_params.n)) = measurement_params.mean;
    if (measurement_params.measure_global)
        y(measurement_params.n) = measurement_params.mean_global;
    Eigen::VectorXd x_post = linear_operator->posterior_mean(xbar, y);
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    x.setZero();
    linear_operator->apply(x_post, f);
    for (int k = 0; k < sampling_params.nwarmup; ++k)
    {
        sampler->apply(f, x);
    };
    Eigen::VectorXd mean(ndof);
    Eigen::VectorXd variance(ndof);
    mean.setZero();
    variance.setZero();
    for (int k = 0; k < sampling_params.nsamples; ++k)
    {
        sampler->apply(f, x);
        mean += (x - mean) / (k + 1.0);
        variance += (x.cwiseProduct(x) - variance) / (k + 1.0);
    }
    std::shared_ptr<VTKWriter> vtk_writer;
    if (measurement_params.dim == 2)
    {
        vtk_writer = std::make_shared<VTKWriter2d>("posterior.vtk", lattice, 1);
    }
    else
    {
        vtk_writer = std::make_shared<VTKWriter3d>("posterior.vtk", lattice, 1);
    }
    vtk_writer->add_state(x_post, "x_post");
    vtk_writer->add_state(mean, "mean");
    vtk_writer->add_state(variance - mean.cwiseProduct(mean), "variance");
    vtk_writer->write();
    if (measurement_params.dim == 2)
    {
        write_vtk_circle(measurement_params.sample_location,
                         measurement_params.radius,
                         "sample_location.vtk");
    }
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
    CholeskyParameters cholesky_params;
    SmootherParameters smoother_params;
    MultigridParameters multigrid_params;
    SamplingParameters sampling_params;
    PriorParameters prior_params;
    ConstantCorrelationLengthModelParameters constantcorrelationlengthmodel_params;
    PeriodicCorrelationLengthModelParameters periodiccorrelationlengthmodel_params;
    MeasurementParameters measurement_params;
    general_params.read_from_file(filename);
    lattice_params.read_from_file(filename);
    cholesky_params.read_from_file(filename);
    smoother_params.read_from_file(filename);
    multigrid_params.read_from_file(filename);
    sampling_params.read_from_file(filename);
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
    std::cout << "Compiled with BLAS/LAPACK support for Eigen." << std::endl;
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
    //   Construct samplers
    /* prepare measurements */
    unsigned int seed = 5418513;
    std::mt19937_64 rng(seed);
    std::shared_ptr<SamplerFactory> presampler_factory = std::make_shared<SORSamplerFactory>(rng,
                                                                                             smoother_params.omega,
                                                                                             forward);
    std::shared_ptr<SamplerFactory> postsampler_factory = std::make_shared<SORSamplerFactory>(rng,
                                                                                              smoother_params.omega,
                                                                                              backward);
    std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory = std::make_shared<IntergridOperatorLinearFactory>();
    std::shared_ptr<SamplerFactory> coarse_sampler_factory;
    if (cholesky_params.factorisation == SparseFactorisation)
    {
        coarse_sampler_factory = std::make_shared<SparseCholeskySamplerFactory>(rng);
    }
    else if (cholesky_params.factorisation == DenseFactorisation)
    {
        coarse_sampler_factory = std::make_shared<DenseCholeskySamplerFactory>(rng);
    }
    std::shared_ptr<Sampler> multigridmc_sampler = std::make_shared<MultigridMCSampler>(posterior_operator,
                                                                                        rng,
                                                                                        multigrid_params,
                                                                                        presampler_factory,
                                                                                        postsampler_factory,
                                                                                        intergrid_operator_factory,
                                                                                        coarse_sampler_factory);
    std::shared_ptr<Sampler> ssor_sampler = std::make_shared<SSORSampler>(posterior_operator, rng, smoother_params.omega);
    std::shared_ptr<Sampler> cholesky_sampler;
    if (general_params.do_cholesky)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        if (cholesky_params.factorisation == SparseFactorisation)
        {
            cholesky_sampler = std::make_shared<SparseCholeskySampler>(posterior_operator, rng, true);
        }
        else if (cholesky_params.factorisation == DenseFactorisation)
        {
            cholesky_sampler = std::make_shared<DenseCholeskySampler>(posterior_operator, rng);
        }
        auto t_finish = std::chrono::high_resolution_clock::now();
        double t_elapsed = std::chrono::duration_cast<std::chrono::seconds>(t_finish - t_start).count();
        std::cout << std::endl;
        std::cout << "time for Cholesky factorisation = " << t_elapsed << " s" << std::endl;
    }

    std::cout << std::endl;
    // Run sampling experiments
    if (general_params.do_cholesky)
    {
        std::cout << "Cholesky" << std::endl;
        measure_sampling_time(cholesky_sampler,
                              sampling_params,
                              measurement_params,
                              "timeseries_cholesky.txt");
        std::cout << std::endl;
    }
    if (general_params.do_ssor)
    {
        std::cout << "SSOR" << std::endl;
        measure_sampling_time(ssor_sampler,
                              sampling_params,
                              measurement_params,
                              "timeseries_ssor.txt");
        std::cout << std::endl;
    }
    if (general_params.do_multigridmc)
    {
        std::cout << "Multigrid MC" << std::endl;
        measure_sampling_time(multigridmc_sampler,
                              sampling_params,
                              measurement_params,
                              "timeseries_multigridmc.txt");
        if (general_params.save_posterior_statistics)
        {
            posterior_statistics(multigridmc_sampler,
                                 sampling_params,
                                 measurement_params);
        }
        std::cout << std::endl;
    }
}
