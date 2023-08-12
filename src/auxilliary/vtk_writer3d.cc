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
    out << "POINT_DATA " << (nx + 1) * (ny + 1) * (nz + 1) << std::endl;
    for (auto it = sample_states.begin(); it != sample_states.end(); ++it)
    {
        std::string label = it->first;
        if (verbose > 0)
            std::cout << "Writing " << label << std::endl;
        Eigen::VectorXd phi = it->second;
        out << "SCALARS " << label << " double 1" << std::endl;
        out << "LOOKUP_TABLE default" << std::endl;
        for (int k = 0; k <= nz; ++k)
        {
            for (int j = 0; j <= ny; ++j)
            {
                for (int i = 0; i <= nx; ++i)
                {
                    double data = 0.0;
                    if ((i > 0) and (i < nx) and (j > 0) and (j < ny) and (k > 0) and (k < nz))
                    {
                        Eigen::VectorXi p(3);
                        p[0] = i;
                        p[1] = j;
                        p[2] = k;
                        data = phi[lattice->vertexidx_euclidean2linear(p)];
                        if (abs(data) < 1.0E-20)
                            data = 0.0;
                    }
                    out << data << std::endl;
                }
            }
        }
    }
    out.close();
}