#include "sampling_time.hh"

/* generate a number of samples, measure runtime and write timeseries to disk */
void measure_sampling_time(std::shared_ptr<Sampler> sampler,
                           const SamplingParameters &sampling_params,
                           const MeasurementParameters &measurement_params,
                           const std::string label,
                           const std::string filename)
{
    const std::shared_ptr<LinearOperator> linear_operator = sampler->get_linear_operator();
    unsigned int ndof = linear_operator->get_ndof();
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    Eigen::VectorXd xbar(ndof);
    xbar.setZero();
    Eigen::VectorXd y(measurement_params.n + measurement_params.measure_global);
    y(Eigen::seqN(0, measurement_params.n)) = measurement_params.mean;
    if (measurement_params.measure_global)
        y(measurement_params.n) = measurement_params.mean_global;
    Eigen::VectorXd mean_x_exact = linear_operator->mean(xbar, y);
    const std::shared_ptr<MeasuredOperator> measured_operator = std::make_shared<MeasuredOperator>(linear_operator,
                                                                                                   measurement_params);
    Eigen::SparseVector<double> sample_vector = measured_operator->measurement_vector(measurement_params.sample_location,
                                                                                      measurement_params.radius);

    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    x.setZero();
    linear_operator->apply(mean_x_exact, f);
    sampler->fix_rhs(f);
    for (int k = 0; k < sampling_params.nwarmup; ++k)
    {
        sampler->apply(f, x);
    };
    std::vector<double> data(sampling_params.nsamples);

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < sampling_params.nsamples; ++k)
    {
        sampler->apply(f, x);
        data[k] = sample_vector.dot(x);
    }
    auto t_finish = std::chrono::high_resolution_clock::now();
    double t_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t_finish - t_start).count() / (1.0 * sampling_params.nsamples);
    printf("  %12s time per sample = %12.4f ms\n", label.c_str(), t_elapsed);
    std::ofstream out;
    out.open(filename);
    for (auto it = data.begin(); it != data.end(); ++it)
        out << *it << std::endl;
    // Compute mean and variance of measurements
    double x_avg = 0.0;
    double xsq_avg = 0.0;
    for (int k = 0; k < sampling_params.nsamples; ++k)
    {
        x_avg += (data[k] - x_avg) / (k + 1.0);
        xsq_avg += (data[k] * data[k] - xsq_avg) / (k + 1.0);
    }
    double variance = xsq_avg - x_avg * x_avg;
    double x_error = sqrt(variance / sampling_params.nsamples);
    double mean_exact, variance_exact;
    linear_operator->observed_mean_and_variance(xbar,
                                                y,
                                                sample_vector,
                                                mean_exact,
                                                variance_exact);
    printf("  %12s mean     = %12.4e +/- %12.4e [ignoring IACT]\n", label.c_str(), x_avg, x_error);
    printf("  %12s mean     = %12.4e\n", "exact", mean_exact);
    printf("  %12s variance = %12.4e\n", label.c_str(), variance);
    printf("  %12s variance = %12.4e\n\n", "exact", variance_exact);

    out.close();
}