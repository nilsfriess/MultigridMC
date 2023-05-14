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
#include "smoother/ssor_smoother.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/diffusion_operator_2d.hh"
#include "intergrid/intergrid_operator_2dlinear.hh"
#include "sampler/cholesky_sampler.hh"
#include "sampler/ssor_sampler.hh"
#include "sampler/multigridmc_sampler.hh"
#include "solver/loop_solver.hh"
#include "preconditioner/multigrid_preconditioner.hh"
#include "auxilliary/parameters.hh"
#include "auxilliary/vtk_writer2d.hh"
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
    const std::shared_ptr<MeasuredDiffusionOperator2d> linear_operator = std::dynamic_pointer_cast<MeasuredDiffusionOperator2d>(sampler->get_linear_operator());
    unsigned int ndof = linear_operator->get_ndof();
    // prior mean (set to zero)
    Eigen::VectorXd xbar(ndof);
    xbar.setZero();
    Eigen::VectorXd y(measurement_params.n + measurement_params.measure_global);
    y(Eigen::seqN(0, measurement_params.n)) = measurement_params.mean;
    if (measurement_params.measure_global)
        y(measurement_params.n) = measurement_params.mean_global;
    Eigen::VectorXd x_post = linear_operator->posterior_mean(xbar, y);
    std::shared_ptr<Lattice2d> lattice = std::dynamic_pointer_cast<Lattice2d>(linear_operator->get_lattice());
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    x.setZero();
    linear_operator->apply(x_post, f);
    for (int k = 0; k < sampling_params.nwarmup; ++k)
    {
        sampler->apply(f, x);
    };
    std::vector<double> data(sampling_params.nsamples);
    Eigen::Vector2i idx;
    idx[0] = int(measurement_params.sample_location[0] * lattice->nx);
    idx[1] = int(measurement_params.sample_location[1] * lattice->nx);
    int j_sample = lattice->idx_euclidean2linear(idx);
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < sampling_params.nsamples; ++k)
    {
        sampler->apply(f, x);
        data[k] = x[j_sample];
    }
    auto t_finish = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_finish - t_start).count() / (1.0 * sampling_params.nsamples);
    std::cout << "  time per sample = " << t_elapsed << " ms" << std::endl;
    std::ofstream out;
    out.open(filename);
    for (auto it = data.begin(); it != data.end(); ++it)
        out << *it << std::endl;

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
    const std::shared_ptr<MeasuredDiffusionOperator2d> linear_operator = std::dynamic_pointer_cast<MeasuredDiffusionOperator2d>(sampler->get_linear_operator());
    unsigned int ndof = linear_operator->get_ndof();

    // prior mean (set to zero)
    Eigen::VectorXd xbar(ndof);
    xbar.setZero();
    Eigen::VectorXd y(measurement_params.n + measurement_params.measure_global);
    y(Eigen::seqN(0, measurement_params.n)) = measurement_params.mean;
    if (measurement_params.measure_global)
        y(measurement_params.n) = measurement_params.mean_global;
    Eigen::VectorXd x_post = linear_operator->posterior_mean(xbar, y);
    std::shared_ptr<Lattice2d> lattice = std::dynamic_pointer_cast<Lattice2d>(linear_operator->get_lattice());
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
    VTKWriter2d vtk_writer("posterior.vtk", Cells, lattice, 1);
    vtk_writer.add_state(x_post, "x_post");
    vtk_writer.add_state(mean, "mean");
    vtk_writer.add_state(variance - mean.cwiseProduct(mean), "variance");
    vtk_writer.write();
    write_vtk_circle(measurement_params.sample_location,
                     0.02,
                     "sample_location.vtk");
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
    MultigridMCParameters multigridmc_params;
    SamplingParameters sampling_params;
    MeasurementParameters measurement_params;
    general_params.read_from_file(filename);
    lattice_params.read_from_file(filename);
    smoother_params.read_from_file(filename);
    multigridmc_params.read_from_file(filename);
    sampling_params.read_from_file(filename);
    measurement_params.read_from_file(filename);

#ifndef NCHOLMOD
    std::cout << "Using Cholesky factorisation from CholMod." << std::endl;
#else  // NCHOLMOD
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << "WARNING: Falling back on Eigen's SimplicalLLT Cholesky factorisation." << std::endl;
    std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << std::endl;
    std::cout << std::endl;
#endif // NCHOLMOD
    // Construct lattice and linear operator
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(lattice_params.nx,
                                                                     lattice_params.ny);
    double alpha_K = 1.5;
    double beta_K = 0.3;
    double alpha_b = 1.2;
    double beta_b = 0.1;
    std::shared_ptr<MeasuredDiffusionOperator2d> linear_operator = std::make_shared<MeasuredDiffusionOperator2d>(lattice,
                                                                                                                 measurement_params.measurement_locations,
                                                                                                                 measurement_params.covariance,
                                                                                                                 measurement_params.measure_global,
                                                                                                                 measurement_params.sigma_global,
                                                                                                                 alpha_K,
                                                                                                                 beta_K,
                                                                                                                 alpha_b,
                                                                                                                 beta_b);
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
    std::shared_ptr<IntergridOperatorFactory> intergrid_operator_factory = std::make_shared<IntergridOperator2dLinearFactory>();
    std::shared_ptr<SamplerFactory> coarse_sampler_factory = std::make_shared<CholeskySamplerFactory>(rng);
    std::shared_ptr<Sampler> multigridmc_sampler = std::make_shared<MultigridMCSampler>(linear_operator,
                                                                                        rng,
                                                                                        multigridmc_params,
                                                                                        presampler_factory,
                                                                                        postsampler_factory,
                                                                                        intergrid_operator_factory,
                                                                                        coarse_sampler_factory);
    std::shared_ptr<Sampler> ssor_sampler = std::make_shared<SSORSampler>(linear_operator, rng, smoother_params.omega);
    std::shared_ptr<Sampler> cholesky_sampler = std::make_shared<CholeskySampler>(linear_operator, rng);
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
        posterior_statistics(multigridmc_sampler,
                             sampling_params,
                             measurement_params);
        std::cout << std::endl;
    }
}
