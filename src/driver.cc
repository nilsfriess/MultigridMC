#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <chrono>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/QR>
#include "libconfig.hh"

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
#include "auxilliary/vtk_writer2d.hh"
#include "auxilliary/statistics.hh"

/** @brief generate a number of samples and write timeseries to disk
 *
 * The field is measured in the centre of the domain
 *
 * @param[in] sampler sampler to be used
 * @param[in] nsamples number of samples
 * @param[in] nwarmup number of warmup samples
 * @param[in] filename name of file to write to
 */
void run(std::shared_ptr<Sampler> sampler,
         const unsigned int nsamples,
         const unsigned int nwarmup,
         const std::string filename)
{
    unsigned int ndof = sampler->get_linear_operator()->get_ndof();
    std::vector<double> data(nsamples);
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    x.setZero();
    f.setZero();

    for (int k = 0; k < nwarmup; ++k)
    {
        sampler->apply(f, x);
    };
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < nsamples; ++k)
    {
        sampler->apply(f, x);
        data[k] = x[ndof / 2];
    }
    auto t_finish = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_finish - t_start).count() / (1.0 * nsamples);
    std::cout << "time per sample = " << t_elapsed << " ms" << std::endl;
    std::ofstream out;
    out.open(filename);
    for (auto it = data.begin(); it != data.end(); ++it)
        out << *it << std::endl;

    out.close();
}
int main(int argc, char *argv[])
{
    libconfig::Config cfg;

    // Read the file. If there is an error, report it and exit.
    try
    {
        cfg.readFile("parameters.cfg");
    }
    catch (const libconfig::FileIOException &fioex)
    {
        std::cerr << "Error while reading configuration file \'parameters.cfg\'." << std::endl;
        return (EXIT_FAILURE);
    }
    unsigned int nx, ny;
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " NX NY" << std::endl;
        exit(-1);
    }
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    unsigned int seed = 1212417;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist_normal(0.0, 1.0);
    std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);
    std::cout << "lattice size : " << nx << " x " << ny << std::endl;
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(nx, ny);
    unsigned int ndof = lattice->M;
    double alpha_K = 1.5;
    double beta_K = 0.3;
    double alpha_b = 1.2;
    double beta_b = 0.1;
    unsigned int n_meas = 4;
    std::cout << "Number of measurements : " << n_meas << std::endl;
    std::vector<Eigen::Vector2d> measurement_locations(n_meas);
    Eigen::MatrixXd Sigma(n_meas, n_meas);
    Sigma.setZero();
    measurement_locations[0] = Eigen::Vector2d({0.25, 0.25});
    measurement_locations[1] = Eigen::Vector2d({0.25, 0.75});
    measurement_locations[2] = Eigen::Vector2d({0.75, 0.25});
    measurement_locations[3] = Eigen::Vector2d({0.75, 0.75});
    for (int k = 0; k < n_meas; ++k)
    {
        Sigma(k, k) = 1.E-6 * (1.0 + 2.0 * dist_uniform(rng));
    }
    // Rotate randomly
    Eigen::MatrixXd A(Eigen::MatrixXd::Random(n_meas, n_meas)), Q;
    A.setRandom();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Q = qr.householderQ();
    Sigma = Q * Sigma * Q.transpose();
    std::shared_ptr<MeasuredDiffusionOperator2d>
        linear_operator = std::make_shared<MeasuredDiffusionOperator2d>(lattice,
                                                                        measurement_locations,
                                                                        Sigma,
                                                                        alpha_K,
                                                                        beta_K,
                                                                        alpha_b,
                                                                        beta_b);

    MultigridMCParameters multigridmc_params;
    multigridmc_params.nlevel = 6;
    multigridmc_params.npresample = 1;
    multigridmc_params.npostsample = 1;
    multigridmc_params.verbose = 1;
    const double omega = 1.0;
    std::shared_ptr<SamplerFactory> presampler_factory = std::make_shared<SORSamplerFactory>(rng,
                                                                                             omega,
                                                                                             forward);
    std::shared_ptr<SamplerFactory> postsampler_factory = std::make_shared<SORSamplerFactory>(rng,
                                                                                              omega,
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
    std::shared_ptr<Sampler> ssor_sampler = std::make_shared<SSORSampler>(linear_operator, rng, omega);
    std::shared_ptr<Sampler> cholesky_sampler = std::make_shared<CholeskySampler>(linear_operator, rng);
    const unsigned int nsamples = 1000;
    const unsigned int nwarmup = 1000;
    std::cout << "Cholesky" << std::endl;
    run(cholesky_sampler, nsamples, nwarmup, "timeseries_cholesky.txt");
    std::cout << std::endl;
    std::cout << "SSOR" << std::endl;
    run(ssor_sampler, nsamples, nwarmup, "timeseries_ssor.txt");
    std::cout << std::endl;
    std::cout << "Multigrid MC" << std::endl;
    run(multigridmc_sampler, nsamples, nwarmup, "timeseries_multigridmc.txt");
    std::cout << std::endl;
}
