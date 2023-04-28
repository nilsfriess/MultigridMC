#include "lattice2d.hh"

/** @file lattice2d.cc
 * @brief Implementation of lattice2d.hh
 */

/* get info string */
std::string Lattice2d::get_info() const
{
    const int buffersize = 80;
    char buffer[buffersize];
    snprintf(buffer, buffersize, "2d lattice, %4d x %4d points, %4d unknowns", nx, ny, M);
    return std::string(buffer);
}