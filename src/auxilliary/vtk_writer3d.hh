#ifndef VTK_WRITER3D_HH
#define VTK_WRITER3D_HH VTK_WRITER3D_HH
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <Eigen/Dense>
#include "vtk_writer.hh"
#include "lattice/lattice3d.hh"

/** @file vtk_writer3d.hh
 *
 * @brief class for writing 3d fields to a vtk files
 */

/** @class VTKWriter3d
 *
 * @brief base class for writing fields to disk
 */
class VTKWriter3d : public VTKWriter
{
public:
    /** @brief Create a new instance
     *
     * @param[in] filename_ name of file to write to
     * @param[in] lattice_ lattice on which data is held
     * @param[in] verbose_ verbosity level
     */
    VTKWriter3d(const std::string filename_,
                const std::shared_ptr<Lattice> lattice_,
                const int verbose_ = 0) : VTKWriter(filename_, verbose_),
                                          lattice(std::dynamic_pointer_cast<Lattice3d>(lattice_)) {}

    /** @brief write all sample states to disk */
    virtual void write() const;

protected:
    /** @brief lattice on which data is held */
    const std::shared_ptr<Lattice3d> lattice;
};

#endif // VTK_WRITER3D_HH