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
        if (verbose > 0)
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

/* write VTK file with circle around a point */
void write_vtk_circle(const Eigen::Vector2d centre,
                      const double radius,
                      const std::string filename)
{
    // Grid specification
    std::ofstream out(filename.c_str());
    out << "# vtk DataFile Version 2.0" << std::endl;
    out << "Sample state" << std::endl;
    out << "ASCII" << std::endl;
    out << "DATASET POLYDATA" << std::endl;
    out << std::endl;
    // number of points used to discretise the circle
    unsigned int npoints = 100;
    double z_offset = 1.E-6;
    out << "POINTS " << npoints << " double" << std::endl;
    for (int j = 0; j < npoints; ++j)
    {
        double x = centre[0] + radius * cos(2 * M_PI * j / (1.0 * npoints)) - 0.5;
        double y = centre[1] + radius * sin(2 * M_PI * j / (1.0 * npoints)) - 0.5;
        out << x << " " << y << " " << z_offset << std::endl;
    }
    out << "POLYGONS 1 " << (npoints + 1) << std::endl;
    out << npoints;
    for (int j = 0; j < npoints; ++j)
    {
        out << " " << j;
    }
    out << std::endl;
}