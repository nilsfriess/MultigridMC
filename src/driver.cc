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

/** @brief structure for general parameters*/
struct GeneralParameters
{
    /** @brief Run the Cholesky sampler? */
    bool do_cholesky;
    /** @brief Run the SSOR sampler? */
    bool do_ssor;
    /** @brief Run the MultigridMC sampler? */
    bool do_multigridmc;
};

/** @brief structure for lattice parameters */
struct LatticeParameters
{
    /** @brief extent in x-direction */
    unsigned int nx;
    /** @brief extent in y-direction */
    unsigned int ny;
};

/** @brief structure for smoother parameters */
struct SmootherParameters
{
    /** @brief overrelaxation factor */
    double omega;
};

/** @brief structure for sampling parameters */
struct SamplingParameters
{
    /** @brief number of samples */
    unsigned int nsamples;
    /** @brief number of warmup samples */
    unsigned int nwarmup;
};

/** @brief read parameters from configuration file
 *
 * @param[in] filename name of file to read
 * @param[out] general_params general parameters
 * @param[out] lattice_params lattice parameters
 * @param[out] smoother_params smoother parameters
 * @param[out] multigridmc_params Multigrid MC parameters
 * @param[out] sampling_params Sampling parameters
 */
int read_configuration_file(const std::string filename,
                            GeneralParameters &general_params,
                            LatticeParameters &lattice_params,
                            SmootherParameters &smoother_params,
                            MultigridMCParameters &multigridmc_params,
                            SamplingParameters &sampling_params)
{
    libconfig::Config cfg;
    std::cout << "==== parameters ====" << std::endl;
    // Read the file. If there is an error, report it and exit.
    try
    {
        cfg.readFile(filename);
    }
    catch (const libconfig::FileIOException &fioex)
    {
        std::cerr << "Error while reading configuration file \'" << filename << "\'." << std::endl;
        return (EXIT_FAILURE);
    }
    const libconfig::Setting &root = cfg.getRoot();
    // General parameters
    try
    {
        const libconfig::Setting &general = root["general"];
        general_params.do_cholesky = general["do_cholesky"];
        general_params.do_ssor = general["do_ssor"];
        general_params.do_multigridmc = general["do_multigridmc"];
    }
    catch (const libconfig::SettingException &ex)
    {
        std::cerr << "Error while reading general configuration from file" << filename << "." << std::endl;
        return (EXIT_FAILURE);
    }
    // Lattice parameters
    try
    {
        const libconfig::Setting &lattice = root["lattice"];
        lattice_params.nx = lattice.lookup("nx");
        lattice_params.ny = lattice.lookup("ny");
        std::cout << "  lattice size = " << lattice_params.nx << " x " << lattice_params.ny << std::endl;
    }
    catch (const libconfig::SettingException &ex)
    {
        std::cerr << "Error while reading lattice configuration from file" << filename << "." << std::endl;
        return (EXIT_FAILURE);
    }
    // Smoother parameters
    try
    {
        const libconfig::Setting &smoother = root["smoother"];
        smoother_params.omega = smoother.lookup("omega");
        std::cout << "  overrelaxation factor = " << smoother_params.omega << std::endl;
    }
    catch (const libconfig::SettingException &ex)
    {
        std::cerr << "Error while reading smoother configuration from file" << filename << "." << std::endl;
        return (EXIT_FAILURE);
    }
    // Multigrid MC parameters
    try
    {
        const libconfig::Setting &multigrid = root["multigridmc"];
        multigridmc_params.nlevel = multigrid.lookup("level");
        multigridmc_params.npresample = multigrid.lookup("npresample");
        multigridmc_params.npostsample = multigrid.lookup("npostsample");
        multigridmc_params.verbose = multigrid.lookup("verbose");
        std::cout << "  MultigridMC levels      = " << multigridmc_params.nlevel << std::endl;
        std::cout << "  MultigridMC npresample  = " << multigridmc_params.npresample << std::endl;
        std::cout << "  MultigridMC nostsample  = " << multigridmc_params.npostsample << std::endl;
    }
    catch (const libconfig::SettingException &ex)
    {
        std::cerr << "Error while reading Multigrid MC configuration from file" << filename << "." << std::endl;
        return (EXIT_FAILURE);
    }
    // Sampling parameters
    try
    {
        const libconfig::Setting &sampling = root["sampling"];
        sampling_params.nsamples = sampling["nsamples"];
        sampling_params.nwarmup = sampling["nwarmup"];
        std::cout << "  number of samples        = " << sampling_params.nsamples << std::endl;
        std::cout << "  number of warmup samples = " << sampling_params.nwarmup << std::endl;
    }
    catch (const libconfig::SettingException &ex)
    {
        std::cerr << "Error while reading sampling configuration from file" << filename << "." << std::endl;
        return (EXIT_FAILURE);
    }
    std::cout << std::endl;
    return 0;
}

/** @brief Construct linear operator
 *
 * @param[in] lattice underlying lattice
 */
std::shared_ptr<MeasuredDiffusionOperator2d> construct_linear_operator(
    const std::shared_ptr<Lattice2d> lattice)
{

    /* prepare measurements */
    unsigned int seed = 1212417;
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist_normal(0.0, 1.0);
    std::uniform_real_distribution<double> dist_uniform(0.0, 1.0);

    unsigned int ndof = lattice->M;
    double alpha_K = 1.5;
    double beta_K = 0.3;
    double alpha_b = 1.2;
    double beta_b = 0.1;
    unsigned int n_meas = 4;
    std::vector<Eigen::Vector2d> measurement_locations(n_meas);
    Eigen::MatrixXd Sigma(n_meas, n_meas);
    Sigma.setZero();
    measurement_locations[0] = Eigen::Vector2d({0.25, 0.25});
    measurement_locations[1] = Eigen::Vector2d({0.25, 0.75});
    measurement_locations[2] = Eigen::Vector2d({0.75, 0.25});
    measurement_locations[3] = Eigen::Vector2d({0.75, 0.75});
    for (int k = 0; k < n_meas; ++k)
    {
        Sigma(k, k) = 1.E-3 * (1.0 + 2.0 * dist_uniform(rng));
    }
    // Rotate randomly
    Eigen::MatrixXd A(Eigen::MatrixXd::Random(n_meas, n_meas)), Q;
    A.setRandom();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(A);
    Q = qr.householderQ();
    Sigma = Q * Sigma * Q.transpose();
    return std::make_shared<MeasuredDiffusionOperator2d>(lattice,
                                                         measurement_locations,
                                                         Sigma,
                                                         alpha_K,
                                                         beta_K,
                                                         alpha_b,
                                                         beta_b);
}

/** @brief generate a number of samples and write timeseries to disk
 *
 * The field is measured in the centre of the domain
 *
 * @param[in] sampler sampler to be used
 * @param[in] sampling_params parameters for sampling
 * @param[in] filename name of file to write to
 */
void run(std::shared_ptr<Sampler> sampler,
         const SamplingParameters sampling_params,
         const std::string filename)
{
    unsigned int ndof = sampler->get_linear_operator()->get_ndof();
    std::vector<double> data(sampling_params.nsamples);
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    x.setZero();
    f.setZero();

    for (int k = 0; k < sampling_params.nwarmup; ++k)
    {
        sampler->apply(f, x);
    };
    auto t_start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < sampling_params.nsamples; ++k)
    {
        sampler->apply(f, x);
        data[k] = x[ndof / 2];
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

/* *********************************************************************** *
 *                                M A I N
 * *********************************************************************** */
int main(int argc, char *argv[])
{
    GeneralParameters general_params;
    LatticeParameters lattice_params;
    SmootherParameters smoother_params;
    MultigridMCParameters multigridmc_params;
    SamplingParameters sampling_params;
    if (argc == 2)
    {
        std::string filename(argv[1]);
        int status = read_configuration_file(filename,
                                             general_params,
                                             lattice_params,
                                             smoother_params,
                                             multigridmc_params,
                                             sampling_params);
        if (status == EXIT_FAILURE)
            exit(-1);
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " CONFIGURATIONFILE" << std::endl;
        exit(-1);
    }
    // Construct lattice and linear operator
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(lattice_params.nx,
                                                                     lattice_params.ny);
    std::shared_ptr<LinearOperator> linear_operator = construct_linear_operator(lattice);
    // Construct samplers
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
        run(cholesky_sampler, sampling_params, "timeseries_cholesky.txt");
        std::cout << std::endl;
    }
    if (general_params.do_ssor)
    {
        std::cout << "SSOR" << std::endl;
        run(ssor_sampler, sampling_params, "timeseries_ssor.txt");
        std::cout << std::endl;
    }
    if (general_params.do_multigridmc)
    {
        std::cout << "Multigrid MC" << std::endl;
        run(multigridmc_sampler, sampling_params, "timeseries_multigridmc.txt");
        std::cout << std::endl;
    }
}
