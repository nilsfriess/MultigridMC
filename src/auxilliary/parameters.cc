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
        exit(-1);
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
        exit(-1);
    }
    return (EXIT_SUCCESS);
}

/* parse general configuration */
void GeneralParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &general = root["general"];
    dim = general["dim"];
    std::cout << "  dimension = " << dim << std::endl;
    do_cholesky = general["do_cholesky"];
    do_ssor = general["do_ssor"];
    do_multigridmc = general["do_multigridmc"];
    save_posterior_statistics = general["save_posterior_statistics"];
}

/* parse lattice configuration  */
void LatticeParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &lattice = root["lattice"];
    nx = lattice.lookup("nx");
    ny = lattice.lookup("ny");
    nz = lattice.lookup("nz");
    std::cout << "  lattice size = " << nx << " x " << ny << " x " << nz << std::endl;
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
    cycle = multigrid.lookup("cycle");
    coarse_scaling = multigrid.lookup("coarse_scaling");
    std::string cycle_label = "";
    if (cycle == 1)
    {
        cycle_label = "( V-cycle )";
    }
    else if (cycle == 2)
    {
        cycle_label = "( W-cycle )";
    }
    verbose = multigrid.lookup("verbose");
    std::cout << "  Multigrid" << std::endl;
    std::cout << "   levels = " << nlevel << std::endl;
    std::cout << "   npresmooth = " << npresmooth << std::endl;
    std::cout << "   npostsmooth = " << npostsmooth << std::endl;
    std::cout << "   cycle = " << cycle << " " << cycle_label << std::endl;
    std::cout << "   coarse_scaling = " << coarse_scaling << std::endl;
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

/* prior configuration */
void PriorParameters::parse_config(const libconfig::Setting &root)
{

    const libconfig::Setting &prior = root["prior"];
    pde_model = prior.lookup("pdemodel").c_str();
    if (not((pde_model == "diffusion") or (pde_model == "shiftedlaplace") or (pde_model == "shiftedbiharmonic")))
    {
        std::cout << "ERROR: Unknown prior: \'" << pde_model << "\'" << std::endl;
        exit(-1);
    }

    correlationlength_model = prior.lookup("correlationlengthmodel").c_str();
    if (not((correlationlength_model == "constant") or (correlationlength_model == "periodic")))
    {
        std::cout << "ERROR: Invalied correlation length model \'" << correlationlength_model << "\'." << std::endl;
        exit(-1);
    }
    std::cout << "  prior" << std::endl;
    std::cout << "    PDEmodel = " << pde_model << std::endl;
    std::cout << "    correlationlengthmodel = " << correlationlength_model << std::endl;
}

/* constant correlation length model configuration */
void ConstantCorrelationLengthModelParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &model = root["constantcorrelationlengthmodel"];
    kappa = model.lookup("kappa");
    std::cout << "  constantcorrelationlengthmodel" << std::endl;
    std::cout << "    kappa = " << kappa << std::endl;
}

/* periodic correlation length model configuration */
void PeriodicCorrelationLengthModelParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &model = root["periodiccorrelationlengthmodel"];
    kappa_min = model.lookup("kappa_min");
    kappa_max = model.lookup("kappa_max");
    std::cout << "  periodiccorrelationlengthmodel" << std::endl;
    std::cout << "    kappa_min = " << kappa_min << std::endl;
    std::cout << "    kappa_max = " << kappa_max << std::endl;
    if (not(kappa_max >= kappa_min))
    {
        std::cout << "ERROR: upper bound on correlation length has to exceed lower bound." << std::endl;
        exit(-1);
    }
    if (not(kappa_min > 0))
    {
        std::cout << "ERROR: lower bound on correlation length has to be positive." << std::endl;
        exit(-1);
    }
}

/* parse measurement configuration */
void MeasurementParameters::parse_config(const libconfig::Setting &root)
{
    const libconfig::Setting &measurements = root["measurements"];
    dim = measurements.lookup("dim");
    unsigned int n_meas = measurements.lookup("n");
    n = n_meas;
    std::cout << "  dimension of measurement locations = " << dim << std::endl;
    std::cout << "  number of measurement points = " << n << std::endl;
    // Measurement locations
    const libconfig::Setting &m_points = measurements.lookup("measurement_locations");
    measurement_locations.clear();
    for (int j = 0; j < n_meas; ++j)
    {
        Eigen::VectorXd v(dim);
        for (int d = 0; d < dim; ++d)
        {
            v[d] = double(m_points[dim * j + d]);
        }
        measurement_locations.push_back(v);
    }
    // Radius
    radius = measurements.lookup("radius");
    std::cout << "  radius of individual measurements = " << radius << std::endl;
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
    Eigen::VectorXd v(dim);
    for (int d = 0; d < dim; ++d)
    {
        v[d] = double(s_point[d]);
    }
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
