#ifndef ACTION_HH
#define ACTION_HH ACTION_HH

#include <memory>
#include <cmath>
#include "lattice2d.hh"
#include "samplestate.hh"

class Action
{
public:
    Action(const Lattice2d lattice2d_) : lattice2d(lattice2d_)
    {
        unsigned int M = lattice2d.M;
        a_centre = new double[M];
        a_north = new double[M];
        a_south = new double[M];
        a_east = new double[M];
        a_west = new double[M];
        unsigned int nx = lattice2d.nx;
        unsigned int ny = lattice2d.ny;
        double hx = 1. / nx;
        double hy = 1. / ny;
        for (unsigned int iy = 0; iy < ny; ++iy)
        {
            for (unsigned int ix = 0; ix < nx; ++ix)
            {
                unsigned int i = nx * iy + ix;
                double K_north = K_diff(ix * hx, (iy + 0.5) * hy);
                double K_south = K_diff(ix * hx, (iy - 0.5) * hy);
                double K_east = K_diff((ix + 0.5) * hx, iy * hy);
                double K_west = K_diff((ix - 0.5) * hx, iy * hy);
                a_centre[i] = 1.0 + (K_east + K_west) / (hx * hx) + (K_north + K_south) / (hy * hy);
                a_north[i] = -K_north / (hy * hy);
                a_south[i] = -K_south / (hy * hy);
                a_east[i] = -K_east / (hy * hy);
                a_west[i] = -K_west / (hy * hy);
            }
        }
    }

    double K_diff(const double x, const double y)
    {
        return 0.8 + 0.25 * sin(2 * M_PI * x) * sin(2 * M_PI * y);
    }

    void apply(const std::shared_ptr<SampleState> X, std::shared_ptr<SampleState> Y)
    /* Apply operator to compute Y = A.X */
    {
        unsigned int nx = lattice2d.nx;
        unsigned int ny = lattice2d.ny;
        for (unsigned int iy = 0; iy < ny; ++iy)
        {
            for (unsigned int ix = 0; ix < nx; ++ix)
            {
                unsigned int i = nx * iy + ix;
                double r = a_centre[i] * X->data[i];
                r += a_north[i] * X->data[nx * ((iy + 1) % ny) + ix];
                r += a_south[i] * X->data[nx * ((iy - 1) % ny) + ix];
                r += a_east[i] * X->data[nx * iy + ((ix + 1) % nx)];
                r += a_west[i] * X->data[nx * iy + ((ix - 1) % nx)];
                Y->data[i] = r;
            }
        }
    }

    void gibbssweep(const std::shared_ptr<SampleState> b, std::shared_ptr<SampleState> X, double omega)
    /* Apply a single Gibbs sweep*/
    {
        unsigned int nx = lattice2d.nx;
        unsigned int ny = lattice2d.ny;

        for (unsigned int iy = 0; iy < ny; ++iy)
        {
            for (unsigned int ix = 0; ix < nx; ++ix)
            {
                unsigned int i = nx * iy + ix;
                double residual = b->data[i];
                residual -= a_north[i] * X->data[nx * ((iy + 1) % ny) + ix];
                residual -= a_south[i] * X->data[nx * ((iy - 1) % ny) + ix];
                residual -= a_east[i] * X->data[nx * iy + ((ix + 1) % nx)];
                residual -= a_west[i] * X->data[nx * iy + ((ix - 1) % nx)];
                X->data[i] = (1. - omega) * X->data[i] + omega / a_centre[i] * residual;
            }
        }
    }

    ~Action()
    {
        delete[] a_centre;
        delete[] a_north;
        delete[] a_south;
        delete[] a_east;
        delete[] a_west;
    }

protected:
    const Lattice2d lattice2d;
    double *a_centre;
    double *a_north;
    double *a_south;
    double *a_east;
    double *a_west;
};

#endif // ACTION_HH
