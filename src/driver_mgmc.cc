#include <iostream>
#include <chrono>
#include <ctime>
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

/** @brief generate a number of samples, measure runtime and write timeseries to disk
 *
 * @param[in] sampler sampler to be used
 * @param[in] sampling_params parameters for sampling
 * @param[in] measurement_params parameters for measurements
 * @param[in] label tag for each output (to simplify parsing later on)
 * @param[in] filename name of file to write to
 */
void measure_sampling_time(std::shared_ptr<Sampler> sampler,
                           const SamplingParameters &sampling_params,
                           const MeasurementParameters &measurement_params,
                           const std::string label,
                           const std::string filename)
{
    const std::shared_ptr<LinearOperator> linear_operator = sampler->get_linear_operator();
    unsigned int ndof = linear_operator->get_ndof();
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    Eigen::VectorXd xbar(ndof);
    xbar.setZero();
    Eigen::VectorXd y(measurement_params.n + measurement_params.measure_global);
    y(Eigen::seqN(0, measurement_params.n)) = measurement_params.mean;
    if (measurement_params.measure_global)
        y(measurement_params.n) = measurement_params.mean_global;
    Eigen::VectorXd mean_x_exact = linear_operator->mean(xbar, y);
    const std::shared_ptr<MeasuredOperator> measured_operator = std::make_shared<MeasuredOperator>(linear_operator,
                                                                                                   measurement_params);
    Eigen::SparseVector<double> sample_vector = measured_operator->measurement_vector(measurement_params.sample_location,
                                                                                      measurement_params.radius);

    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    x.setZero();
    linear_operator->apply(mean_x_exact, f);
    sampler->fix_rhs(f);
    for (int k = 0; k < sampling_params.nwarmup; ++k)
    {
        sampler->apply(f, x);
    };
    std::vector<double> data(sampling_params.nsamples);

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < sampling_params.nsamples; ++k)
    {
        sampler->apply(f, x);
        data[k] = sample_vector.dot(x);
    }
    auto t_finish = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_finish - t_start).count() / (1.0 * sampling_params.nsamples);
    printf("  %12s time per sample = %12.4f ms\n", label.c_str(), t_elapsed);
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
    double mean_exact, variance_exact;
    measured_operator->observed_mean_and_variance(xbar,
                                                  y,
                                                  sample_vector,
                                                  mean_exact,
                                                  variance_exact);
    printf("  %12s mean     = %12.4e +/- %12.4e [ignoring IACT]\n", label.c_str(), x_avg, x_error);
    printf("  %12s mean     = %12.4e\n", "exact", mean_exact);
    printf("  %12s variance = %12.4e\n", label.c_str(), variance);
    printf("  %12s variance = %12.4e\n\n", "exact", variance_exact);

    out.close();
}

/** @brief Compute mean and variance field
 *
 * The mean and variance field are computed with MCMC sampling and
 * written to a vtk file.
 *
 * @param[in] sampler sampler to be used
 * @param[in] sampling_params parameters for sampling
 * @param[in] measurement_params parameters for measurements
 */
void posterior_statistics(std::shared_ptr<Sampler> sampler,
                          const SamplingParameters &sampling_params,
                          const MeasurementParameters &measurement_params)
{
    const std::shared_ptr<LinearOperator> linear_operator = sampler->get_linear_operator();
    unsigned int ndof = linear_operator->get_ndof();

    // prior mean (set to zero)
    Eigen::VectorXd xbar(ndof);
    xbar.setZero();
    Eigen::VectorXd y(measurement_params.n + measurement_params.measure_global);
    y(Eigen::seqN(0, measurement_params.n)) = measurement_params.mean;
    if (measurement_params.measure_global)
        y(measurement_params.n) = measurement_params.mean_global;
    Eigen::VectorXd mean_x_exact = linear_operator->mean(xbar, y);
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    x.setZero();
    linear_operator->apply(mean_x_exact, f);
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
    vtk_writer->add_state(mean, "mean");
    vtk_writer->add_state(variance - mean.cwiseProduct(mean), "variance");
    vtk_writer->add_state(mean_x_exact, "mean_exact");
    vtk_writer->write();
    if (measurement_params.dim == 2)
    {
        write_vtk_circle(measurement_params.sample_location,
                         measurement_params.radius,
                         "sample_location.vtk");
    }
}

/** @brief measure convergence of distribution during warmup
 *
 * Measure the mean mean(k) := E[z^k] and variance var(k) := Var[z^k] of
 * the k-th sample in the chain and look at the decay of mean(k) - E[z] and
 * var(k) - Var[z], where E[z] and Var[z] are the true posterior mean and
 * variance.
 *
 * This subroutine writes a table of the quantities mean(k) - E[z] and var(k) - Var[z]
 * to disk together with error estimators.
 *
 * @param[in] sampler sampler to be used
 * @param[in] sampling_params parameters for sampling
 * @param[in] measurement_params parameters for measurements
 * @param[in] filename name of file with the results
 */
void measure_convergence(std::shared_ptr<Sampler> sampler,
                         const SamplingParameters &sampling_params,
                         const MeasurementParameters &measurement_params,
                         const std::string filename)
{
    const std::shared_ptr<LinearOperator> linear_operator = sampler->get_linear_operator();
    unsigned int ndof = linear_operator->get_ndof();
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    Eigen::VectorXd xbar(ndof);
    xbar.setZero();
    Eigen::VectorXd y(measurement_params.n + measurement_params.measure_global);
    y(Eigen::seqN(0, measurement_params.n)) = measurement_params.mean;
    if (measurement_params.measure_global)
        y(measurement_params.n) = measurement_params.mean_global;
    Eigen::VectorXd mean_x_exact = linear_operator->mean(xbar, y);
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    const std::shared_ptr<MeasuredOperator> measured_operator = std::make_shared<MeasuredOperator>(linear_operator,
                                                                                                   measurement_params);
    Eigen::SparseVector<double> sample_vector = measured_operator->measurement_vector(measurement_params.sample_location,
                                                                                      measurement_params.radius);

    linear_operator->apply(mean_x_exact, f);
    sampler->fix_rhs(f);
    // compute the sample average of (z^k)^alpha for alpha = 1,2,3,4, where z^k is
    // the observed quantity at the k-th step of the chain
    std::vector<double> x_avg(sampling_params.nstepsconvergence + 1, 0.0);
    std::vector<double> x2_avg(sampling_params.nstepsconvergence + 1, 0.0);
    std::vector<double> x3_avg(sampling_params.nstepsconvergence + 1, 0.0);
    std::vector<double> x4_avg(sampling_params.nstepsconvergence + 1, 0.0);
    unsigned int nsamples = sampling_params.nsamplesconvergence;
    for (int k = 0; k < nsamples; ++k)
    {
        x.setZero();
        for (int j = 1; j <= sampling_params.nstepsconvergence; ++j)
        {
            sampler->apply(f, x);
            double z = sample_vector.dot(x);
            x_avg[j] += (z - x_avg[j]) / (k + 1.0);
            x2_avg[j] += (z * z - x2_avg[j]) / (k + 1.0);
            x3_avg[j] += (z * z * z - x3_avg[j]) / (k + 1.0);
            x4_avg[j] += (z * z * z * z - x4_avg[j]) / (k + 1.0);
        }
    }
    // compute exact mean and variance in the stationary chain
    double mean_exact, variance_exact;
    measured_operator->observed_mean_and_variance(xbar,
                                                  y,
                                                  sample_vector,
                                                  mean_exact,
                                                  variance_exact);
    // difference between true mean/variance and sample mean/variance at step k of the chain
    std::vector<double> diff_mean;
    std::vector<double> diff_variance;
    // statistical errors of diff_mean and diff_variance
    std::vector<double> error_diff_mean;
    std::vector<double> error_diff_variance;

    for (int j = 0; j <= sampling_params.nstepsconvergence; ++j)
    {
        diff_mean.push_back(fabs(x_avg[j] - mean_exact));
        diff_variance.push_back(fabs(x2_avg[j] - x_avg[j] * x_avg[j] - variance_exact));
        // unbiased estimator for variance
        double sigma_sq = nsamples / (nsamples - 1.) * (x2_avg[j] - x_avg[j] * x_avg[j]);
        // fourth central moment
        double mu4 = x4_avg[j] - 4 * x_avg[j] * x3_avg[j] + 6 * pow(x_avg[j], 2) * x2_avg[j] - 3 * pow(x_avg[j], 4);
        error_diff_mean.push_back(sqrt(sigma_sq / nsamples));
        error_diff_variance.push_back(sqrt((mu4 - (nsamples - 3.) / (nsamples - 1.) * sigma_sq * sigma_sq) / nsamples));
    }
    std::ofstream out;
    out.open(filename);
    for (int q = 0; q < 2; ++q)
    {
        std::string label;
        if (q == 0)
        {
            out << "**** q_k = |E[z^k] - E[z]| **** " << std::endl;
            label = "mean";
        }
        else
        {
            out << "**** q_k = |Var[z^k] - Var[z]| **** " << std::endl;
            label = "variance";
        }
        char buffer[128];
        sprintf(buffer, "  %12s   %3s : %12s %35s %35s\n", "", "k", "q_k", "q_k/q_0", "q_k/q_{k-1}");
        out << buffer;
        for (int j = 0; j <= sampling_params.nstepsconvergence; ++j)
        {
            double diff;
            double error_diff;
            double diff_0;
            double diff_prev;
            double error_diff_prev;
            if (q == 0)
            {
                diff = diff_mean[j];
                error_diff = error_diff_mean[j];
            }
            else
            {
                diff = diff_variance[j];
                error_diff = error_diff_variance[j];
            }
            if (j == 0)
            {
                diff_0 = diff;
            }
            sprintf(buffer, "  %12s   %3d : %12.8f +/- %12.8f       %12.8f +/- %12.8f      ", label.c_str(), j, diff, error_diff, diff / diff_0, error_diff / diff_0);
            out << buffer;
            if (j > 0)
            {
                double rel_error = diff / diff_prev * sqrt(pow(error_diff / diff, 2) + pow(error_diff_prev / diff_prev, 2));
                sprintf(buffer, " %12.8f +/- %12.8f \n", diff / diff_prev, rel_error);
            }
            else
            {
                sprintf(buffer, " %12s\n", "---");
            }
            out << buffer;
            diff_prev = diff;
            error_diff_prev = error_diff;
        }
        out << std::endl;
    }
    out.close();
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
    auto t_start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(t_start);
    std::cout << std::endl;
    std::cout << "+--------------------------------+" << std::endl;
    std::cout << "! Multigrid Monte Carlo sampling !" << std::endl;
    std::cout << "+--------------------------------+" << std::endl;
    std::cout << std::endl;
    std::cout << "Starting run at " << std::ctime(&start_time) << std::endl;
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
    // Construct samplers
    unsigned int seed = 5418513;
    std::mt19937_64 rng(seed);
    std::shared_ptr<Sampler> multigridmc_sampler = std::make_shared<MultigridMCSampler>(linear_operator,
                                                                                        rng,
                                                                                        multigrid_params,
                                                                                        cholesky_params);
    std::shared_ptr<Sampler> ssor_sampler = std::make_shared<SSORSampler>(linear_operator,
                                                                          rng,
                                                                          smoother_params.omega,
                                                                          smoother_params.nsmooth);
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
        std::cout << "**** Cholesky ****" << std::endl;
        measure_sampling_time(cholesky_sampler,
                              sampling_params,
                              measurement_params,
                              "Cholesky",
                              "timeseries_cholesky.txt");
        std::cout << std::endl;
    }
    if (general_params.do_ssor)
    {
        std::cout << "**** SSOR ****" << std::endl;
        measure_sampling_time(ssor_sampler,
                              sampling_params,
                              measurement_params,
                              "SSOR",
                              "timeseries_ssor.txt");
        measure_convergence(ssor_sampler,
                            sampling_params,
                            measurement_params,
                            "convergence_ssor.txt");
        std::cout << std::endl;
    }
    if (general_params.do_multigridmc)
    {
        std::cout << "**** Multigrid MC ****" << std::endl;
        measure_sampling_time(multigridmc_sampler,
                              sampling_params,
                              measurement_params,
                              "MGMC",
                              "timeseries_multigridmc.txt");
        measure_convergence(multigridmc_sampler,
                            sampling_params,
                            measurement_params,
                            "convergence_multigridmc.txt");
        if (general_params.save_posterior_statistics)
        {
            posterior_statistics(multigridmc_sampler,
                                 sampling_params,
                                 measurement_params);
        }
        std::cout << std::endl;
    }
    // print out total timing information
    auto t_finish = std::chrono::system_clock::now();
    std::time_t finish_time = std::chrono::system_clock::to_time_t(t_finish);
    std::chrono::duration<double> t_diff = t_finish - t_start;
    unsigned int elapsed_seconds = t_diff.count();
    int seconds = elapsed_seconds % 60;
    elapsed_seconds /= 60;
    int minutes = elapsed_seconds % 60;
    elapsed_seconds /= 60;
    int hours = elapsed_seconds % 60;
    std::cout << "Completed run at " << std::ctime(&finish_time);
    printf("Total elapsed time = %3d h %2d m %2d s\n", hours, minutes, seconds);
    std::cout << std::endl;
}
