#ifndef LATTICE2D_HH
#define LATTICE2D_HH LATTICE2D_HH

#include <cassert>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "lattice.hh"

/** @file lattice2d.hh
 *
 * @brief two dimensional lattice
 */

/** @class Lattice2d
 *
 * @brief Two dimensional structured lattice with nx x ny cells
 *
 *  Cells and vectices are arranged ordered lexicographically,
 *  for example for nx = 4, ny =3:
 *
 *  ^ y
 *  !
 *  !
 *
 *  + ------ + ------ + ------ + ------ +
 *  !        !        !        !        !
 *  !    8   !    9   !   10   !   11   !
 *  !        !        !        !        !                 N
 *  + ------ 3 ------ 4 ------ 5 ------ +              W  +  E
 *  !        !        !        !        !                 S
 *  !    4   !    5   !    6   !    7   !
 *  !        !        !        !        !
 *  + ------ 0 ------ 1 ------ 2 ------ +
 *  !        !        !        !        !
 *  !    0   !    1   !    2   !    3   !
 *  !        !        !        !        !
 *  + ------ + ------ + ------ + ------ +  ---> x
 *
 * The cell with index 0 has Euclidean index (0,0)
 * the vertex with index 0 has Euclidean index (1,1)
 */
class Lattice2d : public Lattice
{
public:
  /** @brief Create new instance
   *
   * @param[in] nx_ Extent in x-direction
   * @param[in] ny_ Extent in y-direction
   */
  Lattice2d(const unsigned int nx_, const unsigned int ny_) : nx(nx_),
                                                              ny(ny_),
                                                              hx(1. / double(nx_)),
                                                              hy(1. / double(ny_)),
                                                              Lattice(nx_ * ny_, (nx_ - 1) * (ny_ - 1)) {}

  /** @brief deep copy */
  virtual std::shared_ptr<Lattice> deep_copy()
  {
    return std::make_shared<Lattice2d>(nx, ny);
  };

  /** @brief Convert linear index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi cellidx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell < nx * ny);
    Eigen::VectorXi idx(2);
    idx[0] = ell % nx;
    idx[1] = ell / nx;
    return idx;
  }

  /** @brief Convert Euclidean index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int cellidx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    return idx[1] * nx + idx[0];
  };

  /** @brief Convert linear vertex index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi vertexidx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell < (nx - 1) * (ny - 1));
    Eigen::VectorXi idx(2);
    idx[0] = ell % (nx - 1) + 1;
    idx[1] = ell / (nx - 1) + 1;
    return idx;
  }

  /** @brief Convert Euclidean vertex index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int vertexidx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    assert(idx[0] > 0);
    assert(idx[0] < nx);
    assert(idx[1] > 0);
    assert(idx[1] < ny);
    return (idx[1] - 1) * (nx - 1) + (idx[0] - 1);
  }

  /** @brief Shift a linear cell index by an Euclideanvector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_cellidx(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    assert(ell < nx * ny);
    int i = (ell % nx) + shift[0];
    int j = (ell / nx) + shift[1];
    return j * nx + i;
  }

  /** @brief Shift a linear vertex index by an Euclidean vector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_vertexidx(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    assert(ell < (nx - 1) * (ny - 1));
    int i = (ell % (nx - 1)) + shift[0] + 1;
    int j = (ell / (nx - 1)) + shift[1] + 1;
    assert(i > 0);
    assert(i < nx);
    assert(j > 0);
    assert(j < ny);
    return (j - 1) * (nx - 1) + (i - 1);
  }

  /** @brief Check whether shifting a vertex by an Euclidean vector results in an interior vertex
   *
   * @param[in] ell Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector specifying the corner to inspect, with (0,0,...,0)
   *                  being the lower left corner
   * @param[out] idx_vertex index of vertex, if it is valid (contains garbage otherwise)
   */
  inline virtual bool shifted_vertex_is_internal_vertex(const unsigned int ell, const Eigen::VectorXi shift,
                                                        unsigned int &idx_vertex) const
  {
    assert(ell < (nx - 1) * (ny - 1));
    int i = (ell % (nx - 1)) + shift[0] + 1;
    int j = (ell / (nx - 1)) + shift[1] + 1;
    idx_vertex = (j - 1) * (nx - 1) + (i - 1);
    return ((i > 0) and (i < nx) and (j > 0) and (j < ny));
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
    assert(idx_cell < nx * ny);
    int i = (idx_cell % nx) + corner[0];
    int j = (idx_cell / nx) + corner[1];
    idx_vertex = (j - 1) * (nx - 1) + (i - 1);
    return ((i > 0) and (i < nx) and (j > 0) and (j < ny));
  }

  /** @brief get equivalent index of vertex on next-finer lattice */
  virtual unsigned int fine_vertex_idx(const unsigned int ell) const
  {
    int i = (ell % (nx - 1)) + 1;
    int j = (ell / (nx - 1)) + 1;
    assert(i > 0);
    assert(i < nx);
    assert(j > 0);
    assert(j < ny);
    return (2 * j - 1) * (2 * nx - 1) + (2 * i - 1);
  }

  /** @brief Get coordinates of vertex inside domain [0,1]^d
   *
   * @param[in] ell index of vertex
   */
  virtual Eigen::VectorXd vertex_coordinates(const unsigned int ell) const
  {
    assert(ell < (nx - 1) * (ny - 1));
    Eigen::VectorXd coord(2);
    coord[0] = (ell % (nx - 1) + 1.0) * hx;
    coord[1] = (ell / (nx - 1) + 1.0) * hy;
    return coord;
  }

  /** @brief get coarsened version of lattice */
  virtual std::shared_ptr<Lattice> get_coarse_lattice() const
  {
    if (not((nx % 2 == 0) and (ny % 2 == 0)))
    {
      std::cout << "ERROR: cannot coarsen lattice of size " << nx << " x " << ny;
      std::cout << " [one of the extents is odd]" << std::endl;
      exit(-1);
    }
    if (not((nx / 2 > 1) and (ny / 2 > 1)))
    {
      std::cout << "ERROR: cannot coarsen lattice of size " << nx << " x " << ny;
      std::cout << " [resulting lattice would have no interior vertices]" << std::endl;
      exit(-1);
    }
    return std::make_shared<Lattice2d>(nx / 2, ny / 2);
  }

  /** @brief get info string */
  virtual std::string get_info() const;

  /** @brief return lattice shape */
  inline virtual Eigen::VectorXi shape() const
  {
    Eigen::VectorXi s(2);
    s[0] = nx;
    s[1] = ny;
    return s;
  }

  /** @brief extent in x-direction */
  const unsigned int nx;
  /** @brief extent in y-direction */
  const unsigned int ny;
  /** @brief lattice spacing in x-direction */
  const double hx;
  /** @brief lattice spacing in y-direction */
  const double hy;
};

#endif // LATTICE2D_HH
