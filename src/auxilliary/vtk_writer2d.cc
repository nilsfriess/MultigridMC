/** @file vtk_writer2d.cc
 *
 * @brief implementation of vtk_writer2d.hh
 */
#include "vtk_writer2d.hh"

/** @brief Write sample state to disk */
void VTKWriter2d::write() const
{
    unsigned int nx = lattice->nx;
    unsigned int ny = lattice->ny;
    double hx = 1. / nx;
    double hy = 1. / ny;
    // Grid specification
    std::ofstream out(filename.c_str());
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Sample state" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET STRUCTURED_POINTS" << std::endl;
    out << "DIMENSIONS " << nx + 1 << " " << ny + 1 << " 1 " << std::endl;
    out << "ORIGIN -0.5 -0.5 0.0" << std::endl;
    out << "SPACING " << hx << " " << hy << " 0" << std::endl;
    out << std::endl;
    if (entity == Cells)
    {
        out << "CELL_DATA " << nx * ny << std::endl;
    }
    else
    {
        out << "POINT_DATA " << (nx + 1) * (ny + 1) << std::endl;
    }
    for (auto it = sample_states.begin(); it != sample_states.end(); ++it)
    {
        std::string label = it->first;
        std::cout << "Writing " << label << std::endl;
        Eigen::VectorXd phi = it->second;
        out << "SCALARS " << label << " double 1" << std::endl;
        out << "LOOKUP_TABLE default" << std::endl;
        if (entity == Cells)
        {
            for (unsigned int ell = 0; ell < nx * ny; ++ell)
            {
                out << phi[ell] << std::endl;
            }
        }
        else
        {
            for (int j = 0; j <= ny; ++j)
            {
                for (int i = 0; i <= nx; ++i)
                {
                    unsigned int ell = (j % ny) * nx + (i % nx);
                    out << phi[ell] << std::endl;
                }
            }
        }
    }
    out.close();
}