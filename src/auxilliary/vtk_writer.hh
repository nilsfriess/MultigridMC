#ifndef VTK_WRITER_HH
#define VTK_WRITER_HH VTK_WRITER_HH
#include <string>
#include <iostream>
#include <fstream>
#include <map>
#include <Eigen/Dense>
#include "lattice/lattice.hh"

/** @file vtk_writer.hh
 *
 * @brief classes for writing fields to a vtk files
 */

/** @brief names of grid entities */
enum Entity
{
    Vertices = 0,
    Cells = 1
};

/** @class VTKWriter
 *
 * @brief base class for writing fields to disk
 */
class VTKWriter
{
public:
    /** @brief Create a new instance
     *
     * @param[in] filename_ name of file to write to
     * @param[in] entity_ grid entity which data is associated with
     */
    VTKWriter(const std::string filename_,
              const Entity entity_) : filename(filename_),
                                      entity(entity_) {}

    /** @brief Add state to collection of sample states to be written
     *
     * @param[in] phi state to write to disk
     * @param[in] label label to identify state in file
     */
    void add_state(const Eigen::VectorXd &phi, const std::string label);

    /** @brief write all sample states to disk */
    virtual void write() const = 0;

protected:
    /** @brief name of file to write */
    const std::string filename;
    /** @brief entity associated with data */
    const Entity entity;
    /** @brief dictionary of sample state to be written to disk
     * each state is identified by its label. */
    std::map<std::string, Eigen::VectorXd> sample_states;
};

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
     */
    VTKWriter2d(const std::string filename_,
                const Entity entity_,
                const std::shared_ptr<Lattice2d> lattice_) : VTKWriter(filename_, entity_),
                                                             lattice(lattice_) {}

    /** @brief write all sample states to disk */
    virtual void write() const;

protected:
    /** @brief lattice on which data is held */
    const std::shared_ptr<Lattice2d> lattice;
};

#endif // VTK_WRITER_HH