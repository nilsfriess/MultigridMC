#include "convergence.hh"

/* measure convergence of distribution during warmup */
void measure_convergence(std::shared_ptr<Sampler> sampler,
                         const SamplingParameters &sampling_params,
                         const MeasurementParameters &measurement_params,
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
    unsigned int nsamples = sampling_params.nsamplesconvergence;
    unsigned int nsteps = sampling_params.nstepsconvergence;
    std::vector<double> x_avg(nsteps + 1, 0.0);
    std::vector<double> x2_avg(nsteps + 1, 0.0);
    std::vector<double> x3_avg(nsteps + 1, 0.0);
    std::vector<double> x4_avg(nsteps + 1, 0.0);
#pragma omp parallel num_threads(sampling_params.nthreadsconvergence), default(none), shared(nsamples, nsteps, sample_vector, mean_x_exact, sampler, ndof, linear_operator, x_avg, x2_avg, x3_avg, x4_avg)
    {
        int thread_id = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
#pragma omp master
        printf("Running convergence experiment on %d threads\n", nthreads);
        std::vector<double> x_avg_thread_local(nsteps + 1, 0.0);
        std::vector<double> x2_avg_thread_local(nsteps + 1, 0.0);
        std::vector<double> x3_avg_thread_local(nsteps + 1, 0.0);
        std::vector<double> x4_avg_thread_local(nsteps + 1, 0.0);
        std::shared_ptr<CLCGenerator> thread_local_rng = std::make_shared<CLCGenerator>();
        std::shared_ptr<Sampler> thread_local_sampler = sampler->deep_copy(thread_local_rng);
        Eigen::VectorXd x(ndof);
        Eigen::VectorXd f(ndof);
        linear_operator->apply(mean_x_exact, f);
        thread_local_sampler->fix_rhs(f);
        // compute the sample average of (z^k)^alpha for alpha = 1,2,3,4, where z^k is
        // the observed quantity at the k-th step of the chain
        int k_min = thread_id * (nsamples / nthreads);
        int k_max = (thread_id == nthreads - 1) ? nsamples : (thread_id + 1) * (nsamples / nthreads);
        int nsamples_thread_local = k_max - k_min;
        for (int k = 0; k < nsamples_thread_local; ++k)
        {
            x.setZero();
            for (int j = 0; j <= nsteps; ++j)
            {
                double z = sample_vector.dot(x);
                x_avg_thread_local[j] += (z - x_avg_thread_local[j]) / (k + 1.0);
                x2_avg_thread_local[j] += (z * z - x2_avg_thread_local[j]) / (k + 1.0);
                x3_avg_thread_local[j] += (z * z * z - x3_avg_thread_local[j]) / (k + 1.0);
                x4_avg_thread_local[j] += (z * z * z * z - x4_avg_thread_local[j]) / (k + 1.0);
                if (j < nsteps)
                    thread_local_sampler->apply(f, x);
            }
        }
        // Combine the thread-local averages into a global average
        for (int j = 0; j <= nsteps; ++j)
        {
            // weighting factor of thread-local sum
            double rho = nsamples_thread_local / double(nsamples);
#pragma omp atomic
            x_avg[j] += rho * x_avg_thread_local[j];
#pragma omp atomic
            x2_avg[j] += rho * x2_avg_thread_local[j];
#pragma omp atomic
            x3_avg[j] += rho * x3_avg_thread_local[j];
#pragma omp atomic
            x4_avg[j] += rho * x4_avg_thread_local[j];
        }
    }
    // compute exact mean and variance in the stationary chain
    double mean_exact, variance_exact;
    linear_operator->observed_mean_and_variance(xbar,
                                                y,
                                                sample_vector,
                                                mean_exact,
                                                variance_exact);
    // difference between true mean/variance and sample mean/variance at step k of the chain
    std::vector<double> diff_mean;
    std::vector<double> diff_variance;
    // statistical errors of diff_mean and diff_variance
    std::vector<double> error_diff_mean;
    std::vector<double> error_diff_variance;

    for (int j = 0; j <= sampling_params.nstepsconvergence; ++j)
    {
        diff_mean.push_back(fabs(x_avg[j] - mean_exact));
        diff_variance.push_back(fabs(x2_avg[j] - x_avg[j] * x_avg[j] - variance_exact));
        // unbiased estimator for variance
        double sigma_sq = nsamples / (nsamples - 1.) * (x2_avg[j] - x_avg[j] * x_avg[j]);
        // fourth central moment
        double mu4 = x4_avg[j] - 4 * x_avg[j] * x3_avg[j] + 6 * pow(x_avg[j], 2) * x2_avg[j] - 3 * pow(x_avg[j], 4);
        error_diff_mean.push_back(sqrt(sigma_sq / nsamples));
        error_diff_variance.push_back(sqrt((mu4 - (nsamples - 3.) / (nsamples - 1.) * sigma_sq * sigma_sq) / nsamples));
    }
    std::ofstream out;
    out.open(filename);
    for (int q = 0; q < 2; ++q)
    {
        std::string label;
        if (q == 0)
        {
            out << "**** q_k = |E[z^k] - E[z]| **** " << std::endl;
            label = "mean";
        }
        else
        {
            out << "**** q_k = |Var[z^k] - Var[z]| **** " << std::endl;
            label = "variance";
        }
        char buffer[256];
        sprintf(buffer, "  %12s   %3s : %12s %35s %35s\n", "", "k", "q_k", "q_k/q_0", "q_k/q_{k-1}");
        out << buffer;
        for (int j = 0; j <= sampling_params.nstepsconvergence; ++j)
        {
            double diff;
            double error_diff;
            double diff_0;
            double diff_prev;
            double error_diff_prev;
            if (q == 0)
            {
                diff = diff_mean[j];
                error_diff = error_diff_mean[j];
            }
            else
            {
                diff = diff_variance[j];
                error_diff = error_diff_variance[j];
            }
            if (j == 0)
            {
                diff_0 = diff;
            }
            sprintf(buffer, "  %12s   %3d : %12.8f +/- %12.8f       %12.8f +/- %12.8f      ", label.c_str(), j, diff, error_diff, diff / diff_0, error_diff / diff_0);
            out << buffer;
            if (j > 0)
            {
                double rel_error = diff / diff_prev * sqrt(pow(error_diff / diff, 2) + pow(error_diff_prev / diff_prev, 2));
                sprintf(buffer, " %12.8f +/- %12.8f \n", diff / diff_prev, rel_error);
            }
            else
            {
                sprintf(buffer, " %12s\n", "---");
            }
            out << buffer;
            diff_prev = diff;
            error_diff_prev = error_diff;
        }
        out << std::endl;
    }
    out.close();
}