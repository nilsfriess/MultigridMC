#ifndef POSTERIOR_STATISTICS
#define POSTERIOR_STATISTICS POSTERIOR_STATISTICS

#include <memory>
#include <vector>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "auxilliary/parameters.hh"
#include "sampler/sampler.hh"
#include "auxilliary/vtk_writer2d.hh"
#include "auxilliary/vtk_writer3d.hh"

/** @file posterior_statistics.hh */

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
                          const MeasurementParameters &measurement_params);

#endif // POSTERIOR_STATISTICS