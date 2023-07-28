#include "lattice1d.hh"

/** @file lattice1d.cc
 * @brief Implementation of lattice1d.hh
 */

/* get info string */
std::string Lattice1d::get_info() const
{
    const int buffersize = 80;
    char buffer[buffersize];
    snprintf(buffer, buffersize, "1d lattice, %4d points, %4d unknowns", n, Nvertex);
    return std::string(buffer);
}