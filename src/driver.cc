#include <iostream>
#include <chrono>
#include <random>
#include "lattice.hh"
#include "samplestate.hh"
#include "action.hh"
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
    std::shared_ptr<Lattice2d> lattice2d = std::make_shared<Lattice2d>(nx, ny);
    unsigned int seed = 1212417;
    std::mt19937_64 rng(seed);
    std::shared_ptr<Lattice2d> coarse_lattice = std::static_pointer_cast<Lattice2d>(lattice2d->get_coarse_lattice());
    std::shared_ptr<DiffusionOperator2d> action = std::make_shared<DiffusionOperator2d>(lattice2d, rng);
    std::shared_ptr<DiffusionOperator2d> coarse_action = std::make_shared<DiffusionOperator2d>(coarse_lattice, rng);
    IntergridOperator2dAvg intergrid_operator_avg(lattice2d);
    intergrid_operator_avg.coarsen_operator(action, coarse_action);
    std::cout << lattice2d->M << std::endl;
    std::shared_ptr<SampleState> X = std::make_shared<SampleState>(lattice2d->M);
    std::shared_ptr<SampleState> Y = std::make_shared<SampleState>(lattice2d->M);
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (unsigned int i = 0; i < lattice2d->M; ++i)
    {
        X->data[i] = dist(mt);
    }

    /* Measure applications of operator */
    std::cout << "==== operator application ====" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    unsigned int niter = 1000;
    for (unsigned int k = 0; k < niter; ++k)
    {
        action->apply(X, Y);
    }

    auto t_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_finish - t_start;
    std::cout << " elapsed = " << t_elapsed.count() << " s ";
    std::cout << " [ " << niter << " iterations ] " << std::endl;
    std::cout << " time per application = " << 1E6 * t_elapsed.count() / niter << " mu s" << std::endl;

    /* Measure applications of smooth */
    std::cout << "==== smoother application ====" << std::endl;
    t_start = std::chrono::high_resolution_clock::now();
    for (unsigned int k = 0; k < niter; ++k)
    {
        action->gibbssweep(Y, X);
    }

    t_finish = std::chrono::high_resolution_clock::now();
    t_elapsed = t_finish - t_start;
    std::cout << " elapsed = " << t_elapsed.count() << " s ";
    std::cout << " [ " << niter << " iterations ] " << std::endl;
    std::cout << " time per application = " << 1E6 * t_elapsed.count() / niter << " mu s" << std::endl;
}