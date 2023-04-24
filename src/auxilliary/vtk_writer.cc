/** @file vtk_writer.cc
 *
 * @brief implementation of vtk_writer.hh
 */
#include "vtk_writer.hh"

/** @brief Add state to collection of sample states to be written */
void VTKWriter::add_state(const Eigen::VectorXd &phi, const std::string label)
{
    sample_states[label] = phi;
}
