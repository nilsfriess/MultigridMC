
#ifndef SAMPLING_TIME_HH
#define SAMPLING_TIME_HH SAMPLING_TIME_HH

#include <memory>
#include <chrono>
#include <iostream>
#include <fstream>
#include "auxilliary/parameters.hh"
#include "sampler/sampler.hh"
#include "linear_operator/linear_operator.hh"
#include "linear_operator/measured_operator.hh"

/** @file sampling_time.hh */

/** @brief generate a number of samples, measure runtime and write timeseries to disk
 *
 * @param[in] sampler sampler to be used
 * @param[in] sampling_params parameters for sampling
 * @param[in] measurement_params parameters for measurements
 * @param[in] label tag for each output (to simplify parsing later on)
 * @param[in] filename name of file to write to
 */
void measure_sampling_time(std::shared_ptr<Sampler> sampler,
                           const SamplingParameters &sampling_params,
                           const MeasurementParameters &measurement_params,
                           const std::string label,
                           const std::string filename);

#endif // SAMPLING_TIME_HH