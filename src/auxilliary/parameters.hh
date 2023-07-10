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
    int read_from_file(const std::string filename);

    /** @brief constructor */
    Parameters() = default;

    /** @brief destructor */
    virtual ~Parameters() = default;

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
    virtual void parse_config(const libconfig::Setting &root);

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
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief extent in x-direction */
    unsigned int nx;
    /** @brief extent in y-direction */
    unsigned int ny;
};

/** @brief type of Cholesky factorisation */
enum cholesky_t
{
    SparseFactorisation = 0, // Sparse factorisation
    DenseFactorisation = 1   // Dense factorisation
};

/** @struct Cholesky factorisation parameters */
class CholeskyParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief Cholesky factorisation to use */
    cholesky_t factorisation;
};

/** @brief structure for smoother parameters */
class SmootherParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief overrelaxation factor */
    double omega;
};

/** @brief iterative linear solver parameters
 */
class IterativeSolverParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief relative tolerance for solving */
    double rtol;
    /** @brief absolute tolerance for solving */
    double atol;
    /** @brief maximum number of iterations */
    unsigned int maxiter;
    /** @brief verbosity level */
    int verbose;
};

/** @struct multigrid parameters */
class MultigridParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief Number of levels */
    unsigned int nlevel;
    /** @brief Number of presmoothing steps */
    unsigned int npresmooth;
    /** @brief number of postsmoothing steps */
    unsigned int npostsmooth;
};

/** @struct Multigrid Monte Carlo parameters */
class MultigridMCParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

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
    virtual void parse_config(const libconfig::Setting &root);

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
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief number of measurements */
    unsigned int n;
    /** @brief measurement locations */
    std::vector<Eigen::Vector2d> measurement_locations;
    /** @brief measured averages */
    Eigen::VectorXd mean;
    /** @brief covariance matrix of measurements */
    Eigen::MatrixXd covariance;
    /** @brief ignore cross-correlations in masurements (i.e. use only the diagonal
     *  of the covariance matrix)? */
    bool ignore_measurement_cross_correlations;
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