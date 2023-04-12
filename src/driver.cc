#include <iostream>
#include <chrono>
#include <random>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include "lattice.hh"
#include "sampler.hh"
#include "diffusion_operator_2d.hh"
#include "intergrid_operator.hh"

int main(int argc, char *argv[])
{
    unsigned int nx, ny;
    if (argc != 3)
    {
        std::cout << "Usage: " << argv[0] << " NX NY" << std::endl;
        exit(-1);
    }
    nx = atoi(argv[1]);
    ny = atoi(argv[2]);
    std::cout << "lattice size : " << nx << " x " << ny << std::endl;
    std::shared_ptr<Lattice2d> lattice = std::make_shared<Lattice2d>(nx, ny);
    unsigned int seed = 1212417;
    std::mt19937_64 rng(seed);
    std::shared_ptr<Lattice2d> coarse_lattice = std::static_pointer_cast<Lattice2d>(lattice->get_coarse_lattice());
    DiffusionOperator2d linear_operator = DiffusionOperator2d(lattice);
    GibbsSampler Sampler(linear_operator, rng);
    IntergridOperator2dAvg intergrid_operator_avg(lattice);
    LinearOperator coarse_operator = intergrid_operator_avg.coarsen_operator(linear_operator);
    std::cout << lattice->M << std::endl;
    Eigen::VectorXd X(lattice->M);
    Eigen::VectorXd Y(lattice->M);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (unsigned int i = 0; i < lattice->M; ++i)
    {
        X[i] = dist(mt);
    }
    Eigen::VectorXd X_coarse(coarse_lattice->M);
    intergrid_operator_avg.restrict(X, X_coarse);

    /* Measure applications of operator */
    std::cout << "==== operator application ====" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    unsigned int niter = 1000;
    for (unsigned int k = 0; k < niter; ++k)
    {
        linear_operator.apply(X, Y);
    }

    auto t_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_finish - t_start;
    std::cout << " elapsed = " << t_elapsed.count() << " s ";
    std::cout << " [ " << niter << " iterations ] " << std::endl;
    std::cout << " time per application = " << 1E6 * t_elapsed.count() / niter << " mu s" << std::endl;

    /* Measure applications of sparse eigen matrix*/
    Eigen::SparseMatrix<double> A_sparse = linear_operator.as_sparse();
    std::cout << "==== Eigen::sparse application ====" << std::endl;
    t_start = std::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < niter; ++k)
    {
        Y = A_sparse * X;
    }

    t_finish = std::chrono::high_resolution_clock::now();
    t_elapsed = t_finish - t_start;
    std::cout << " elapsed = " << t_elapsed.count() << " s ";
    std::cout << " [ " << niter << " iterations ] " << std::endl;
    std::cout << " time per application = " << 1E6 * t_elapsed.count() / niter << " mu s" << std::endl;

    /* Measure applications of smooth */
    std::cout << "==== Sampler application ====" << std::endl;
    t_start = std::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < niter; ++k)
    {
        // linear_operator->gibbssweep(Y, X);
        Sampler.apply(Y, X);
    }

    t_finish = std::chrono::high_resolution_clock::now();
    t_elapsed = t_finish - t_start;
    std::cout << " elapsed = " << t_elapsed.count() << " s ";
    std::cout << " [ " << niter << " iterations ] " << std::endl;
    std::cout << " time per application = " << 1E6 * t_elapsed.count() / niter << " mu s" << std::endl;
}