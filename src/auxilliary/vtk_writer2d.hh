#ifndef VTK_WRITER2D_HH
#define VTK_WRITER2D_HH VTK_WRITER2D_HH
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <Eigen/Dense>
#include "vtk_writer.hh"
#include "lattice/lattice2d.hh"

/** @file vtk_writer2d.hh
 *
 * @brief class for writing 2d fields to a vtk files
 */

/** @class VTKWriter2d
 *
 * @brief base class for writing fields to disk
 */
class VTKWriter2d : public VTKWriter
{
public:
    /** @brief Create a new instance
     *
     * @param[in] filename_ name of file to write to
     * @param[in] entity_ grid entity which data is associated with
     * @param[in] lattice_ lattice on which data is held
     * @param[in] verbose_ verbosity level
     */
    VTKWriter2d(const std::string filename_,
                const Entity entity_,
                const std::shared_ptr<Lattice2d> lattice_,
                const int verbose_ = 0) : VTKWriter(filename_, entity_, verbose_),
                                          lattice(lattice_) {}

    /** @brief write all sample states to disk */
    virtual void write() const;

protected:
    /** @brief lattice on which data is held */
    const std::shared_ptr<Lattice2d> lattice;
};

/** @brief write VTK file with circle around a point
 *
 * @param[in] centre centre of circle
 * @param[in] radius radius of circle
 * @param[in] filename name of vtk file to write
 */
void write_vtk_circle(const Eigen::Vector2d centre,
                      const double radius,
                      const std::string filename);

#endif // VTK_WRITER2D_HH