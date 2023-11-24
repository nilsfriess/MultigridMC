#include "posterior_statistics.hh"

/* @brief Compute mean and variance field */
void posterior_statistics(std::shared_ptr<Sampler> sampler,
                          const SamplingParameters &sampling_params,
                          const MeasurementParameters &measurement_params)
{
    const std::shared_ptr<LinearOperator> linear_operator = sampler->get_linear_operator();
    unsigned int ndof = linear_operator->get_ndof();

    // prior mean (set to zero)
    Eigen::VectorXd xbar(ndof);
    xbar.setZero();
    Eigen::VectorXd y(measurement_params.n + measurement_params.measure_global);
    y(Eigen::seqN(0, measurement_params.n)) = measurement_params.mean;
    if (measurement_params.measure_global)
        y(measurement_params.n) = measurement_params.mean_global;
    Eigen::VectorXd mean_x_exact = linear_operator->mean(xbar, y);
    std::shared_ptr<Lattice> lattice = linear_operator->get_lattice();
    Eigen::VectorXd x(ndof);
    Eigen::VectorXd f(ndof);
    x.setZero();
    linear_operator->apply(mean_x_exact, f);
    for (int k = 0; k < sampling_params.nwarmup; ++k)
    {
        sampler->apply(f, x);
    };
    Eigen::VectorXd mean(ndof);
    Eigen::VectorXd variance(ndof);
    mean.setZero();
    variance.setZero();
    for (int k = 0; k < sampling_params.nsamples; ++k)
    {
        sampler->apply(f, x);
        mean += (x - mean) / (k + 1.0);
        variance += (x.cwiseProduct(x) - variance) / (k + 1.0);
    }
    std::shared_ptr<VTKWriter> vtk_writer;
    if (measurement_params.dim == 2)
    {
        vtk_writer = std::make_shared<VTKWriter2d>("posterior.vtk", lattice, 1);
    }
    else
    {
        vtk_writer = std::make_shared<VTKWriter3d>("posterior.vtk", lattice, 1);
    }
    vtk_writer->add_state(mean, "mean");
    vtk_writer->add_state(variance - mean.cwiseProduct(mean), "variance");
    vtk_writer->add_state(mean_x_exact, "mean_exact");
    vtk_writer->write();
    if (measurement_params.dim == 2)
    {
        write_vtk_circle(measurement_params.sample_location,
                         measurement_params.radius,
                         "sample_location.vtk");
    }
}