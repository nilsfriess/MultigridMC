#include "parameters.hh"

/** @file parameters.cc
 *
 * @brief implementation of parameters.hh
 */

#include <vector>
#include <iostream>
#include <string>
#include <typeinfo>
#include <Eigen/Dense>
#include "libconfig.hh"

/** @file parameters.cc
 *
 * @brief implementation of parameters.hh
 * */

/* read parameters from disk */
int Parameters::read_from_file(const std::string filename)
{
    std::string classname = typeid(*this).name();
    libconfig::Config cfg;
    // Read the file. If there is an error, report it and exit.
    try
    {
        cfg.readFile(filename.c_str());
    }
    catch (const libconfig::FileIOException &fioex)
    {
        std::cerr << "Error in class \'" << classname << "\': "
                  << "cannot open configuration file \'" << filename << "\'." << std::endl;
        return (EXIT_FAILURE);
    }
    const libconfig::Setting &root = cfg.getRoot();
    try
    {
        parse_config(root);
    }

    catch (const libconfig::SettingException &ex)
    {
        std::cerr << "Error in class \'" << classname << "\': "
                  << "cannot read configuration from file \'" << filename << "\'." << std::endl;
        return (EXIT_FAILURE);
    }
    return (EXIT_SUCCESS);
}

/* parse general configuration */
void GeneralParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &general = root["general"];
    do_cholesky = general["do_cholesky"];
    do_ssor = general["do_ssor"];
    do_multigridmc = general["do_multigridmc"];
}

/* parse lattice configuration  */
void LatticeParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &lattice = root["lattice"];
    nx = lattice.lookup("nx");
    ny = lattice.lookup("ny");
    std::cout << "  lattice size = " << nx << " x " << ny << std::endl;
}

/* parse cholesky configuration */
void CholeskyParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &cholesky = root["cholesky"];
    std::string fac_str = cholesky.lookup("factorisation");
    if (fac_str == "sparse")
    {
        factorisation = SparseFactorisation;
    }
    else if (fac_str == "dense")
    {
        factorisation = DenseFactorisation;
    }
    else
    {
        std::cout << "ERROR: Unknown Cholesky factorisation: \'" << fac_str << "\'" << std::endl;
        exit(-1);
    }
    std::cout << "  Cholesky factorisation = " << fac_str << std::endl;
}

/* parse smoother configuration */
void SmootherParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &smoother = root["smoother"];
    omega = smoother.lookup("omega");
    std::cout << "  overrelaxation factor = " << omega << std::endl;
}

/* parse linear solver configuration */
void IterativeSolverParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &iterative_solver = root["iterative_solver"];
    rtol = iterative_solver.lookup("rtol");
    atol = iterative_solver.lookup("atol");
    maxiter = iterative_solver.lookup("maxiter");
    verbose = iterative_solver.lookup("verbose");
    std::cout << "   relative tolerance = " << rtol << std::endl;
    std::cout << "   absolute tolerance = " << atol << std::endl;
    std::cout << "   maxiter = " << maxiter << std::endl;
}

/* parse multigrid configuration */
void MultigridParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &multigrid = root["multigrid"];
    nlevel = multigrid.lookup("nlevel");
    npresmooth = multigrid.lookup("npresmooth");
    npostsmooth = multigrid.lookup("npostsmooth");
    std::cout << "   multigrid levels = " << nlevel << std::endl;
    std::cout << "   npresmooth = " << npresmooth << std::endl;
    std::cout << "   npostsmooth = " << npostsmooth << std::endl;
}

/* parse multigrid Monte Carlo configuration */
void MultigridMCParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &multigrid = root["multigridmc"];
    nlevel = multigrid.lookup("level");
    npresample = multigrid.lookup("npresample");
    npostsample = multigrid.lookup("npostsample");
    verbose = multigrid.lookup("verbose");
    std::cout << "  MultigridMC levels      = " << nlevel << std::endl;
    std::cout << "  MultigridMC npresample  = " << npresample << std::endl;
    std::cout << "  MultigridMC npostsample  = " << npostsample << std::endl;
}

/* parse sampling configuration */
void SamplingParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &sampling = root["sampling"];
    nsamples = sampling["nsamples"];
    nwarmup = sampling["nwarmup"];
    std::cout << "  number of samples        = " << nsamples << std::endl;
    std::cout << "  number of warmup samples = " << nwarmup << std::endl;
}

/* parse measurement configuration */
void MeasurementParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &measurements = root["measurements"];
    unsigned int n_meas = measurements.lookup("n");
    n = n_meas;
    std::cout << "  number of measurement points = " << n << std::endl;
    // Measurement locations
    const libconfig::Setting &m_points = measurements.lookup("measurement_locations");
    measurement_locations.clear();
    for (int j = 0; j < n_meas; ++j)
    {
        Eigen::Vector2d v;
        v(0) = double(m_points[2 * j]);
        v(1) = double(m_points[2 * j + 1]);
        measurement_locations.push_back(v);
    }
    // Measured averages
    const libconfig::Setting &s_mean = measurements.lookup("mean");
    mean = Eigen::VectorXd(n_meas);
    for (int j = 0; j < n_meas; ++j)
    {
        mean(j) = double(s_mean[j]);
    }
    // Covariance matrix
    const libconfig::Setting &Sigma = measurements.lookup("covariance");
    covariance = Eigen::MatrixXd(n_meas, n_meas);
    for (int j = 0; j < n_meas; ++j)
    {
        for (int k = 0; k < n_meas; ++k)
        {
            covariance(j, k) = Sigma[j + n_meas * k];
        }
    }
    // Ignore cross-correlations in measurements?
    ignore_measurement_cross_correlations = measurements.lookup("ignore_measurement_cross_correlations");
    std::cout << "  ignore correlations between measurements? ";
    if (ignore_measurement_cross_correlations)
    {
        std::cout << "yes" << std::endl;
    }
    else
    {
        std::cout << "no" << std::endl;
    }
    // Sample location
    const libconfig::Setting &s_point = measurements.lookup("sample_location");
    Eigen::Vector2d v;
    v(0) = double(s_point[0]);
    v(1) = double(s_point[1]);
    sample_location = v;
    measure_global = measurements.lookup("measure_global");
    sigma_global = measurements.lookup("sigma_global");
    mean_global = measurements.lookup("mean_global");
    std::cout << "  measure global average across domain? ";
    if (measure_global)
    {
        std::cout << "yes" << std::endl;
        std::cout << "  mean of global average = " << mean_global << std::endl;
        std::cout << "  variance of global average = " << sigma_global << std::endl;
    }
    else
    {
        std::cout << "no" << std::endl;
    }
    std::cout << std::endl;
}
