/** @file vtk_writer3d.cc
 *
 * @brief implementation of vtk_writer3d.hh
 */
#include "vtk_writer3d.hh"

/** @brief Write sample state to disk */
void VTKWriter3d::write() const
{
    unsigned int nx = lattice->nx;
    unsigned int ny = lattice->ny;
    unsigned int nz = lattice->nz;
    double hx = 1. / nx;
    double hy = 1. / ny;
    double hz = 1. / nz;
    // Grid specification
    std::ofstream out(filename.c_str());
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Sample state" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET STRUCTURED_POINTS" << std::endl;
    out << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " " << nz + 1 << std::endl;
    out << "ORIGIN -0.5 -0.5 -5.0" << std::endl;
    out << "SPACING " << hx << " " << hy << " " << hz << std::endl;
    out << std::endl;
    if (entity == Cells)
    {
        out << "CELL_DATA " << nx * ny * nz << std::endl;
    }
    else
    {
        out << "POINT_DATA " << (nx + 1) * (ny + 1) * (nz + 1) << std::endl;
    }
    for (auto it = sample_states.begin(); it != sample_states.end(); ++it)
    {
        std::string label = it->first;
        if (verbose > 0)
            std::cout << "Writing " << label << std::endl;
        Eigen::VectorXd phi = it->second;
        out << "SCALARS " << label << " double 1" << std::endl;
        out << "LOOKUP_TABLE default" << std::endl;
        if (entity == Cells)
        {
            for (unsigned int ell = 0; ell < nx * ny * nz; ++ell)
            {
                out << phi[ell] << std::endl;
            }
        }
        else
        {
            for (int k = 0; k <= nz; ++k)
            {
                for (int j = 0; j <= ny; ++j)
                {
                    for (int i = 0; i <= nx; ++i)
                    {
                        unsigned int ell = (k % nz) * nx * ny + (j % ny) * nx + (i % nx);
                        out << phi[ell] << std::endl;
                    }
                }
            }
        }
    }
    out.close();
}