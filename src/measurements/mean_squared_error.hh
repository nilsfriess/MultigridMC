#ifndef MEAN_SQUARED_ERROR_HH
#define MEAN_SQUARED_ERROR_HH MEAN_SQUARED_ERROR_HH

#include <memory>
#include <fstream>
#include <vector>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "auxilliary/parameters.hh"
#include "auxilliary/statistics.hh"
#include "lattice/lattice.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/measured_operator.hh"
#include "sampler/sampler.hh"

/** @file mean_squared_error.hh */

/** @brief measure mean squared error
 *
 * Consider an observable z and let hat(z)_M = 1/M sum_{k=0}^{M-1} z^{(k)}
 * be an estimator for this variable obtained with the Markov Chain
 * z^{(0)}, z^{(1)}, ..., z^{(M-1)} This code computes an estimator for
 * the mean squared error which is given by
 *
 * E[ |E[z] - hat(z)_M|^2 ].
 *
 * To achieve this, a large number of chains is run.
 *
 * @param[in] sampler sampler to be used
 * @param[in] sampling_params parameters for sampling
 * @param[in] measurement_params parameters for measurements
 * @param[in] filename name of file with the results
 */
void measure_mean_squared_error(std::shared_ptr<Sampler> sampler,
                                const SamplingParameters &sampling_params,
                                const MeasurementParameters &measurement_params,
                                const std::string filename);

#endif // MEAN_SQUARED_ERROR_HH