#include "mean_squared_error.hh"

/* measure convergence of mean squared error */
void measure_mean_squared_error(std::shared_ptr<Sampler> sampler,
                                const SamplingParameters &sampling_params,
                                const MeasurementParameters &measurement_params,
                                const std::string filename)
{
    std::cout << "  Measuring MSE" << std::endl;
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
    unsigned int nsamples = sampling_params.nsamplesmse;
    unsigned int nsteps = sampling_params.nstepsmse;
    std::vector<double> x_avg(nsteps, 0.0);
    std::vector<double> x2_avg(nsteps, 0.0);
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    linear_operator->apply(mean_x_exact, f);
    sampler->fix_rhs(f);
    // compute exact mean and variance in the stationary chain
    double mean_exact, variance_exact;
    linear_operator->observed_mean_and_variance(xbar,
                                                y,
                                                sample_vector,
                                                mean_exact,
                                                variance_exact);

    for (int k = 0; k < nsamples; ++k)
    {
        x.setZero();
        double z = 0;
        for (int j = 0; j < nsteps; ++j)
        {
            z += (sample_vector.dot(x) - z) / (j + 1);
            double mse = (z - mean_exact) * (z - mean_exact);
            x_avg[j] += (mse - x_avg[j]) / (k + 1.0);
            x2_avg[j] += (mse * mse - x2_avg[j]) / (k + 1.0);
            sampler->apply(f, x);
        }
    }

    std::ofstream out;
    out.open(filename);
    for (int j = 0; j < nsteps; ++j)
    {
        double mse = x_avg[j];
        double mse_error = 1. / (nsamples - 1.) * (x2_avg[j] - x_avg[j] * x_avg[j]);
        char buffer[256];
        sprintf(buffer, " %5d : %12.6e +/- %12.6e\n", j, mse, mse_error);
        out << buffer;
    }
    out.close();
}