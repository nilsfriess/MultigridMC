#ifndef LATTICE_HH
#define LATTICE_HH LATTICE_HH
#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <Eigen/Dense>

/** @file lattice.hh
 *
 * @brief Provides base class for lattices
 */

/** @class Lattice
 *
 * @brief Abstract base class for lattices
 */
class Lattice
{
public:
  /** @brief Create a new instance
   *
   * @param[in] Ncell_ number of lattice cells
   * @param[in] Nvertex_ number of interior vertices
   */
  Lattice(const unsigned int Ncell_,
          const unsigned int Nvertex_) : Ncell(Ncell_),
                                         Nvertex(Nvertex_) {}

  /** @brief deep copy */
  virtual std::shared_ptr<Lattice> deep_copy() = 0;

  /** @brief cell volume */
  double cell_volume() const
  {
    Eigen::VectorXi s = shape();
    double volume = 1.0;
    for (int d = 0; d < dim(); ++d)
    {
      volume /= s[d];
    }
    return volume;
  }

  /** @brief Convert linear cell index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi cellidx_linear2euclidean(const unsigned int ell) const = 0;

  /** @brief Convert Euclidean cell index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int cellidx_euclidean2linear(const Eigen::VectorXi idx) const = 0;

  /** @brief Convert linear vertex index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi vertexidx_linear2euclidean(const unsigned int ell) const = 0;

  /** @brief Convert Euclidean vertex index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int vertexidx_euclidean2linear(const Eigen::VectorXi idx) const = 0;

  /** @brief Shift a linear cell index by an Euclideanvector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_cellidx(const unsigned int ell, const Eigen::VectorXi shift) const = 0;

  /** @brief Shift a linear vertex index by an Euclidean vector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_vertexidx(const unsigned int ell, const Eigen::VectorXi shift) const = 0;

  /** @brief Check whether shifting a vertex by an Euclidean vector results in an interior vertex
   *
   * @param[in] ell Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector specifying the corner to inspect, with (0,0,...,0)
   *                  being the lower left corner
   * @param[out] idx_vertex index of vertex, if it is valid (contains garbage otherwise)
   */
  inline virtual bool shifted_vertex_is_internal_vertex(const unsigned int ell, const Eigen::VectorXi shift,
                                                        unsigned int &idx_vertex) const = 0;

  /** @brief Check whether a specific vertex of a cell with given index is an internal vertex
   *
   * Returns the index of the vertex, if the test has been successful
   *
   * @param[in] idx_cell index of cell
   * @param[in] corner Euclidean shift vector specifying the corner to inspect, with (0,0,...,0)
   *                   being the lower left corner
   * @param[out] idx_vertex index of vertex, if it is valid (contains garbage otherwise)
   */
  inline virtual bool corner_is_internal_vertex(const unsigned int idx_cell,
                                                const Eigen::VectorXi corner,
                                                unsigned int &idx_vertex) const = 0;

  /** @brief get equivalent index of vertex on next-finer lattice */
  virtual unsigned int fine_vertex_idx(const unsigned int ell) const = 0;

  /** @brief Get coordinates of vertex inside domain [0,1]^d
   *
   * @param[in] ell index of vertex
   */
  virtual Eigen::VectorXd vertex_coordinates(const unsigned int ell) const = 0;

  /** @brief return lattice shape */
  inline virtual Eigen::VectorXi shape() const = 0;

  /** @brief return lattice dimension */
  inline virtual int dim() const { return shape().size(); }

  /** @brief get coarsened version of lattice */
  virtual std::shared_ptr<Lattice> get_coarse_lattice() const = 0;

  /** @brief get info string */
  virtual std::string get_info() const = 0;

  /** @brief total number of cells in lattice */
  const unsigned int Ncell;

  /** @brief total number of interior vertices of lattice */
  const unsigned int Nvertex;
};

#endif // LATTICE_HH