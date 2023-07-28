#ifndef LATTICE1D_HH
#define LATTICE1D_HH LATTICE1D_HH
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "lattice.hh"

/** @file lattice1d.hh
 *
 * @brief one dimensional lattice
 */

/** @class Lattice1d
 *
 * @brief One dimensional structured lattice with n cells
 *
 * For example, if n = 4:
 *
 *        0       1       2       3        (cells)
 *    + ----- 0 ----- 1 ----- 2 ----- x    (vertices)
 *
 * The cell with index 0 has Euclidean index (0,)
 * the vertex with index 0 has Euclidean index (1,)
 */
class Lattice1d : public Lattice
{
public:
  /** @brief Create new instance
   *
   * @param[in] n_ Extent of lattice (= number of cells)
   */
  Lattice1d(const unsigned int n_) : n(n_), Lattice(n_, n_ - 1) {}

  /** @brief Convert linear cell index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi cellidx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell < Ncell);
    Eigen::VectorXi idx(1);
    idx[0] = ell;
    return idx;
  }

  /** @brief Convert Euclidean cell index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int cellidx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    assert(idx[0] >= 0);
    assert(idx[0] < n);
    return idx[0];
  }

  /** @brief Convert linear vertex index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi vertexidx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell >= 0);
    assert(ell < Ncell - 1);
    Eigen::VectorXi idx(1);
    idx[0] = ell + 1;
    return idx;
  }

  /** @brief Convert Euclidean vertex index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int vertexidx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    assert(idx[0] > 0);
    assert(idx[0] < n);
    return idx[0] - 1;
  }

  /** @brief Shift a linear cell index by an Euclidean vector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_cellidx(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    int i = ell + shift[0];
    assert(i >= 0);
    assert(i < n);
    return i;
  }

  /** @brief Shift a linear vertex index by an Euclidean vector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_vertexidx(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    int i = ell + shift[0];
    assert(i >= 0);
    assert(i < n - 1);
    return i;
  }

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
                                                unsigned int &idx_vertex) const
  {
    assert(idx_cell < n);
    idx_vertex = idx_cell + corner[0] - 1;
    return ((idx_vertex > 0) and (idx_vertex < n));
  }

  /** @brief get equivalent index of vertex on next-finer lattice */
  virtual unsigned int fine_vertex_idx(const unsigned int ell) const
  {
    return 2 * ell + 1;
  }

  /** @brief get coarsened version of lattice */
  virtual std::shared_ptr<Lattice> get_coarse_lattice() const
  {
    if (not(n % 2 == 0))
    {
      std::cout << "ERROR: cannot coarsen lattice of size " << n;
      std::cout << " [extent is odd]" << std::endl;
      exit(-1);
    }
    if (not(n / 2 > 1))
    {
      std::cout << "ERROR: cannot coarsen lattice of size " << n;
      std::cout << " [resulting lattice would have no interior vertices]" << std::endl;
      exit(-1);
    }
    return std::make_shared<Lattice1d>(n / 2);
  }

  /** @brief get info string */
  virtual std::string get_info() const;

  /** @brief return lattice shape */
  inline virtual Eigen::VectorXi shape() const
  {
    Eigen::VectorXi s(1);
    s[0] = n;
    return s;
  }

  /** @brief extent of lattice */
  const unsigned int n;
};

#endif // LATTICE1D_HH