#ifndef PARAMETERS_HH
#define PARAMETERS_HH PARAMETERS_HH
#include <vector>
#include <iostream>
#include <string>
#include <typeinfo>
#include <Eigen/Dense>
#include "libconfig.hh"

/** @file parameters.hh
 *
 * @brief read parameters from configuration files
 * */

/** @brief Base class for parameters */
class Parameters
{
public:
    /** @brief read parameters from disk
     *
     * @param[in] filename name of configuration file to read parameters from
     */
    int read_from_file(const std::string filename)
    {
        std::string classname = typeid(*this).name();
        libconfig::Config cfg;
        // Read the file. If there is an error, report it and exit.
        try
        {
            cfg.readFile(filename);
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

protected:
    /** @brief parse configuration
     *
     * This virtual method has to be implemented by each derived class
     * to actually parse the parameters in the libconfig configuration object.
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root) = 0;
};

/** @brief structure for general parameters*/
class GeneralParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root)
    {
        const libconfig::Setting &general = root["general"];
        do_cholesky = general["do_cholesky"];
        do_ssor = general["do_ssor"];
        do_multigridmc = general["do_multigridmc"];
    }
    /** @brief Run the Cholesky sampler? */
    bool do_cholesky;
    /** @brief Run the SSOR sampler? */
    bool do_ssor;
    /** @brief Run the MultigridMC sampler? */
    bool do_multigridmc;
};

/** @brief structure for lattice parameters */
class LatticeParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root)
    {
        const libconfig::Setting &lattice = root["lattice"];
        nx = lattice.lookup("nx");
        ny = lattice.lookup("ny");
        std::cout << "  lattice size = " << nx << " x " << ny << std::endl;
    }
    /** @brief extent in x-direction */
    unsigned int nx;
    /** @brief extent in y-direction */
    unsigned int ny;
};

/** @brief structure for smoother parameters */
class SmootherParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root)
    {
        const libconfig::Setting &smoother = root["smoother"];
        omega = smoother.lookup("omega");
        std::cout << "  overrelaxation factor = " << omega << std::endl;
    }

    /** @brief overrelaxation factor */
    double omega;
};

/** @struct Multigrid Monte Carlo parameters */
class MultigridMCParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root)
    {
        const libconfig::Setting &multigrid = root["multigridmc"];
        nlevel = multigrid.lookup("level");
        npresample = multigrid.lookup("npresample");
        npostsample = multigrid.lookup("npostsample");
        verbose = multigrid.lookup("verbose");
        std::cout << "  MultigridMC levels      = " << nlevel << std::endl;
        std::cout << "  MultigridMC npresample  = " << npresample << std::endl;
        std::cout << "  MultigridMC nostsample  = " << npostsample << std::endl;
    }
    /** @brief Number of levels */
    unsigned int nlevel;
    /** @brief Number of presmoothing steps */
    unsigned int npresample;
    /** @brief number of postsmoothing steps */
    unsigned int npostsample;
    /** @brief verbosity level */
    int verbose;
};

/** @brief structure for sampling parameters */
class SamplingParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root)
    {
        const libconfig::Setting &sampling = root["sampling"];
        nsamples = sampling["nsamples"];
        nwarmup = sampling["nwarmup"];
        std::cout << "  number of samples        = " << nsamples << std::endl;
        std::cout << "  number of warmup samples = " << nwarmup << std::endl;
    }
    /** @brief number of samples */
    unsigned int nsamples;
    /** @brief number of warmup samples */
    unsigned int nwarmup;
};

/** @brief Structure for measurement parameters */
class MeasurementParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root)
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
    /** @brief number of measurements */
    unsigned int n;
    /** @brief measurement locations */
    std::vector<Eigen::Vector2d> measurement_locations;
    /** @brief measured averages */
    Eigen::VectorXd mean;
    /** @brief covariance matrix of measurements */
    Eigen::MatrixXd covariance;
    /** @brief sample location */
    Eigen::Vector2d sample_location;
    /** @brief measure global average of field? */
    bool measure_global;
    /** @brief variance of global average */
    double sigma_global;
    /** @brief mean of global average */
    double mean_global;
};

#endif // PARAMETERS_HH