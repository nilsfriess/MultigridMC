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
#include "linear_operator/diffusion_operator.hh"
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
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    x.setZero();
    f.setRandom();
    for (int k = 0; k < sampling_params.nwarmup; ++k)
    {
        sampler->apply(f, x);
    };
    std::vector<double> data(sampling_params.nsamples);
    Eigen::VectorXi idx(lattice->dim());
    Eigen::VectorXi shape = lattice->shape();
    for (int d = 0; d < lattice->dim(); ++d)
    {
        idx[d] = int(measurement_params.sample_location[d] * shape[d]);
    }
    int j_sample = lattice->vertexidx_euclidean2linear(idx);
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
        vtk_writer = std::make_shared<VTKWriter2d>("posterior.vtk", Cells, lattice, 1);
    }
    else
    {
        vtk_writer = std::make_shared<VTKWriter3d>("posterior.vtk", Cells, lattice, 1);
    }
    vtk_writer->add_state(x_post, "x_post");
    vtk_writer->add_state(mean, "mean");
    vtk_writer->add_state(variance - mean.cwiseProduct(mean), "variance");
    vtk_writer->write();
    if (measurement_params.dim == 2)
    {
        write_vtk_circle(measurement_params.sample_location,
                         0.02,
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
    MultigridMCParameters multigridmc_params;
    SamplingParameters sampling_params;
    DiffusionParameters diffusion_params;
    MeasurementParameters measurement_params;
    general_params.read_from_file(filename);
    lattice_params.read_from_file(filename);
    cholesky_params.read_from_file(filename);
    smoother_params.read_from_file(filename);
    multigridmc_params.read_from_file(filename);
    sampling_params.read_from_file(filename);
    diffusion_params.read_from_file(filename);
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
    std::shared_ptr<DiffusionOperator> diffusion_operator;
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
    diffusion_operator = std::make_shared<DiffusionOperator>(lattice,
                                                             diffusion_params.alpha_K,
                                                             diffusion_params.beta_K,
                                                             diffusion_params.alpha_b,
                                                             diffusion_params.beta_b);
    std::shared_ptr<MeasuredOperator> linear_operator = std::make_shared<MeasuredOperator>(diffusion_operator,
                                                                                           measurement_params.measurement_locations,
                                                                                           measurement_params.covariance,
                                                                                           measurement_params.ignore_measurement_cross_correlations,
                                                                                           measurement_params.measure_global,
                                                                                           measurement_params.sigma_global);
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
    std::shared_ptr<Sampler> multigridmc_sampler = std::make_shared<MultigridMCSampler>(linear_operator,
                                                                                        rng,
                                                                                        multigridmc_params,
                                                                                        presampler_factory,
                                                                                        postsampler_factory,
                                                                                        intergrid_operator_factory,
                                                                                        coarse_sampler_factory);
    std::shared_ptr<Sampler> ssor_sampler = std::make_shared<SSORSampler>(linear_operator, rng, smoother_params.omega);
    std::shared_ptr<Sampler> cholesky_sampler;
    if (general_params.do_cholesky)
    {
        auto t_start = std::chrono::high_resolution_clock::now();
        if (cholesky_params.factorisation == SparseFactorisation)
        {
            cholesky_sampler = std::make_shared<SparseCholeskySampler>(linear_operator, rng, true);
        }
        else if (cholesky_params.factorisation == DenseFactorisation)
        {
            cholesky_sampler = std::make_shared<DenseCholeskySampler>(linear_operator, rng);
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
