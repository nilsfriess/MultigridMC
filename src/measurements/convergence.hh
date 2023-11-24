#ifndef CONVERGENCE_HH
#define CONVERGENCE_HH CONVERGENCE_HH

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

/** @file convergence.hh */

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
                         const std::string filename);

#endif // CONVERGENCE_HH