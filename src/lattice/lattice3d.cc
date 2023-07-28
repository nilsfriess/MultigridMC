#include "lattice3d.hh"

/** @file lattice3d.cc
 * @brief Implementation of lattice3d.hh
 */

/* get info string */
std::string Lattice3d::get_info() const
{
    const int buffersize = 80;
    char buffer[buffersize];
    snprintf(buffer, buffersize, "3d lattice, %4d x %4d x %4d points, %4d unknowns", nx, ny, nz, Nvertex);
    return std::string(buffer);
}