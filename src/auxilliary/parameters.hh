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

    /** @brief underlying prior to use */
    std::string prior;
    /** @brief spatial dimension */
    int dim;
    /** @brief Run the Cholesky sampler? */
    bool do_cholesky;
    /** @brief Run the SSOR sampler? */
    bool do_ssor;
    /** @brief Run the MultigridMC sampler? */
    bool do_multigridmc;
    /** @brief save posterior statistics to disk? */
    bool save_posterior_statistics;
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
    /** @brief extent in z-direction */
    unsigned int nz;
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
    /** @brief cycle type (1 = V-cycle, 2 = W-cycle)*/
    unsigned int cycle;
    /** @briaf factor with which to scale to coarse grid correction */
    double coarse_scaling;
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

/** @brief structure for diffusion
 *
 * The coefficient is assumed to be of the form
 *
 *    K(x,y) = alpha_K + beta_K * sin(2 pi x) * sin(2 pi y)                 [in 2d]
 * or
 *    K(x,y) = alpha_K + beta_K * sin(2 pi x) * sin(2 pi y) * sin(2 pi z)   [in 3d]
 *
 * and the zero order term is assumed to be
 *
 *    b(x,y) = alpha_b + beta_b * cos(2 pi x) * cos(2 pi y)                 [in 2d]
 * or
 *    b(x,y) = alpha_b + beta_b * cos(2 pi x) * cos(2 pi y) * cos(2 pi z)   [in 3d]
 *
 */
class DiffusionParameters : public Parameters
{
public:
    /** @brief parse configuration
     *
     * @param[in] root root of configuration object
     */
    virtual void parse_config(const libconfig::Setting &root);

    /** @brief constant alpha_K in diffusion coefficient K(x,y) */
    double alpha_K;
    /** @brief constant beta_K in diffusion coefficient K(x,y) */
    double beta_K;
    /** @brief constant alpha_b in zero-order term b(x,y) */
    double alpha_b;
    /** @brief constant beta_K in zero-order term b(x,y) */
    double beta_b;
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

    /** @brief dimension of measurements */
    int dim;
    /** @brief number of measurements */
    unsigned int n;
    /** @brief measurement locations */
    std::vector<Eigen::VectorXd> measurement_locations;
    /** @brief radius of measurement function */
    double radius;
    /** @brief measured averages */
    Eigen::VectorXd mean;
    /** @brief covariance matrix of measurements */
    Eigen::MatrixXd covariance;
    /** @brief ignore cross-correlations in masurements (i.e. use only the diagonal
     *  of the covariance matrix)? */
    bool ignore_measurement_cross_correlations;
    /** @brief sample location */
    Eigen::VectorXd sample_location;
    /** @brief measure global average of field? */
    bool measure_global;
    /** @brief variance of global average */
    double sigma_global;
    /** @brief mean of global average */
    double mean_global;
};

#endif // PARAMETERS_HH