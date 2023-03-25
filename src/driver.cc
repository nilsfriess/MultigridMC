#include <iostream>
#include <chrono>
#include <random>
#include "lattice2d.hh"
#include "samplestate.hh"
#include "action.hh"

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
    Lattice2d lattice2d(nx, ny);
    Action action(lattice2d);
    std::shared_ptr<SampleState> X = std::make_shared<SampleState>(lattice2d.M);
    std::shared_ptr<SampleState> Y = std::make_shared<SampleState>(lattice2d.M);

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (unsigned int i = 0; i < lattice2d.M; ++i)
    {
        X->data[i] = dist(mt);
    }

    /* Measure applications of operator */
    std::cout << "==== operator application ====" << std::endl;
    auto t_start = std::chrono::high_resolution_clock::now();
    unsigned int niter = 1000;
    for (unsigned int k = 0; k < niter; ++k)
    {
        action.apply(X, Y);
    }

    auto t_finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> t_elapsed = t_finish - t_start;
    std::cout << " elapsed = " << t_elapsed.count() << " s ";
    std::cout << " [ " << niter << " iterations ] " << std::endl;
    std::cout << " time per application = " << 1E6 * t_elapsed.count() / niter << " mu s" << std::endl;

    /* Measure applications of smooth */
    std::cout << "==== smoother application ====" << std::endl;
    t_start = std::chrono::high_resolution_clock::now();
    double omega = 0.95;
    for (unsigned int k = 0; k < niter; ++k)
    {
        action.smooth(X, Y, omega);
    }

    t_finish = std::chrono::high_resolution_clock::now();
    t_elapsed = t_finish - t_start;
    std::cout << " elapsed = " << t_elapsed.count() << " s ";
    std::cout << " [ " << niter << " iterations ] " << std::endl;
    std::cout << " time per application = " << 1E6 * t_elapsed.count() / niter << " mu s" << std::endl;
}