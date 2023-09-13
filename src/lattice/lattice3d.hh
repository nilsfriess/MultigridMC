#ifndef LATTICE3D_HH
#define LATTICE3D_HH LATTICE3D_HH

#include <cassert>
#include <memory>
#include <iostream>
#include <Eigen/Dense>
#include "lattice.hh"

/** @file lattice3d.hh
 *
 * @brief three dimensional lattice
 */

/** @class Lattice3d
 *
 * @brief Three dimensional structured lattice with nx x ny x nz cells
 *
 *  Points are arranged ordered lexicographically, for example for nx = 4, ny = 3, nz = 3
 *  (cell indices are projected onto the facets below):
 *
 *
 *  ^ y
 *  !
 *  !             layer z=0                                             layer z=1
 *
 *  + ------ + ------ + ------ + ------ +                   + ------ + ------ + ------ + ------ +
 *  !        !        !        !        !                   !        !        !        !        !
 *  !    8   !    9   !   10   !   11   !                   !   20   !   21   !   22   !   23   !
 *  !        !        !        !        !          N        !        !        !        !        !
 *  + ------ + ------ + ------ + ------ +       W  +  E     + ------ 3 ------ 4 ------ 5 ------ +
 *  !        !        !        !        !          S        !        !        !        !        !
 *  !    4   !    5   !    6   !    7   !                   !   16   !   17   !   18   !   19   !
 *  !        !        !        !        !                   !        !        !        !        !
 *  + ------ + ------ + ------ + ------ +                   + ------ 0 ------ 1 ------ 2 ------ +
 *  !        !        !        !        !                   !        !        !        !        !
 *  !    0   !    1   !    2   !    3   !                   !   12   !   13   !   14   !   15   !
 *  !        !        !        !        !                   !        !        !        !        !
 *  + ------ + ------ + ------ + ------ +  ---> x           + ------ + ------ + ------ + ------ +
 *
 *                layer z=2                                             layer z=3
 *
 *  + ------ + ------ + ------ + ------ +                   + ------ + ------ + ------ + ------ +
 *  !        !        !        !        !                   !        !        !        !        !
 *  !   32   !   33   !   34   !   35   !                   !        !        !        !        !
 *  !        !        !        !        !                   !        !        !        !        !
 *  + ------ 9 ----- 10 ----- 11 ------ +                   + ------ + ------ + ------ + ------ +
 *  !        !        !        !        !                   !        !        !        !        !
 *  !   28   !   29   !   30   !   31   !                   !        !        !        !        !
 *  !        !        !        !        !                   !        !        !        !        !
 *  + ------ 6 ------ 7 ------ 8 ------ +                   + ------ + ------ + ------ + ------ +
 *  !        !        !        !        !                   !        !        !        !        !
 *  !   24   !   25   !   26   !   27   !                   !        !        !        !        !
 *  !        !        !        !        !                   !        !        !        !        !
 *  + ------ + ------ + ------ + ------ +                   + ------ + ------ + ------ + ------ +
 *
 * The cell with index 0 has Euclidean index (0,0,0)
 * the vertex with index 0 has Euclidean index (1,1,1)
 */
class Lattice3d : public Lattice
{
public:
  /** @brief Create new instance
   *
   * @param[in] nx_ Extent in x-direction
   * @param[in] ny_ Extent in y-direction
   * @param[in] nz_ Extent in z-direction
   */
  Lattice3d(const unsigned int nx_,
            const unsigned int ny_,
            const unsigned int nz_) : nx(nx_),
                                      ny(ny_),
                                      nz(nz_),
                                      hx(1. / double(nx_)),
                                      hy(1. / double(ny_)),
                                      hz(1. / double(nz_)),
                                      Lattice(nx_ * ny_ * nz_, (nx_ - 1) * (ny_ - 1) * (nz_ - 1)) {}

  /** @brief deep copy */
  virtual std::shared_ptr<Lattice> deep_copy()
  {
    return std::make_shared<Lattice3d>(nx, ny, nz);
  };

  /** @brief Convert linear index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi cellidx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell < Ncell);
    Eigen::VectorXi idx(3);
    idx[0] = (ell % (nx * ny)) % nx;
    idx[1] = (ell % (nx * ny)) / nx;
    idx[2] = ell / (nx * ny);
    return idx;
  }

  /** @brief Convert Euclidean index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int cellidx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    assert(idx[0] >= 0);
    assert(idx[0] < nx);
    assert(idx[1] >= 0);
    assert(idx[1] < ny);
    assert(idx[2] >= 0);
    assert(idx[2] < nz);
    return idx[2] * nx * ny + idx[1] * nx + idx[0];
  };

  /** @brief Convert linear vertex index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi vertexidx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell < (nx - 1) * (ny - 1) * (nz - 1));
    Eigen::VectorXi idx(3);
    idx[0] = (ell % ((nx - 1) * (ny - 1))) % (nx - 1) + 1;
    idx[1] = (ell % ((nx - 1) * (ny - 1))) / (nx - 1) + 1;
    idx[2] = ell / ((nx - 1) * (ny - 1)) + 1;
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
    assert(idx[2] > 0);
    assert(idx[2] < nz);
    return (idx[2] - 1) * (nx - 1) * (ny - 1) + (idx[1] - 1) * (nx - 1) + (idx[0] - 1);
  }

  /** @brief Shift a linear index by an Euclideanvector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_cellidx(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    assert(ell >= 0);
    assert(ell < nx * ny * nz);
    int i = (ell % (nx * ny)) % nx + shift[0];
    int j = (ell % (nx * ny)) / nx + shift[1];
    int k = ell / (nx * ny) + shift[2];
    assert(i >= 0);
    assert(i < nx);
    assert(j >= 0);
    assert(j < ny);
    assert(k >= 0);
    assert(k < nz);
    return k * nx * ny + j * nx + i;
  }

  /** @brief Shift a linear vertex index by an Euclidean vector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_vertexidx(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    assert(ell < (nx - 1) * (ny - 1) * (nz - 1));
    int i = (ell % ((nx - 1) * (ny - 1))) % (nx - 1) + shift[0] + 1;
    int j = (ell % ((nx - 1) * (ny - 1))) / (nx - 1) + shift[1] + 1;
    int k = ell / ((nx - 1) * (ny - 1)) + shift[2] + 1;
    assert(i > 0);
    assert(i < nx);
    assert(j > 0);
    assert(j < ny);
    assert(k > 0);
    assert(k < nz);
    return (k - 1) * (nx - 1) * (ny - 1) + (j - 1) * (nx - 1) + (i - 1);
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
    assert(ell < (nx - 1) * (ny - 1) * (nz - 1));
    int i = (ell % ((nx - 1) * (ny - 1))) % (nx - 1) + shift[0] + 1;
    int j = (ell % ((nx - 1) * (ny - 1))) / (nx - 1) + shift[1] + 1;
    int k = ell / ((nx - 1) * (ny - 1)) + shift[2] + 1;
    idx_vertex = (k - 1) * (nx - 1) * (ny - 1) + (j - 1) * (nx - 1) + (i - 1);
    return ((i > 0) and (i < nx) and (j > 0) and (j < ny) and (k > 0) and (k < nz));
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
    assert(idx_cell < nx * ny * nz);
    int i = (idx_cell % (nx * ny)) % nx + corner[0];
    int j = (idx_cell % (nx * ny)) / nx + corner[1];
    int k = idx_cell / (nx * ny) + corner[2];
    idx_vertex = (k - 1) * (nx - 1) * (ny - 1) + (j - 1) * (nx - 1) + (i - 1);
    return ((i > 0) and (i < nx) and (j > 0) and (j < ny) and (k > 0) and (k < nz));
  }

  /** @brief get equivalent index of vertex on next-finer lattice */
  virtual unsigned int fine_vertex_idx(const unsigned int ell) const
  {
    assert(ell < (nx - 1) * (ny - 1) * (nz - 1));
    int i = (ell % ((nx - 1) * (ny - 1))) % (nx - 1) + 1;
    int j = (ell % ((nx - 1) * (ny - 1))) / (nx - 1) + 1;
    int k = ell / ((nx - 1) * (ny - 1)) + 1;
    return (2 * k - 1) * (2 * nx - 1) * (2 * ny - 1) + (2 * j - 1) * (2 * nx - 1) + 2 * i - 1;
  }

  /** @brief Get coordinates of vertex inside domain [0,1]^d
   *
   * @param[in] ell index of vertex
   */
  virtual Eigen::VectorXd vertex_coordinates(const unsigned int ell) const
  {
    assert(ell < (nx - 1) * (ny - 1) * (nz - 1));
    Eigen::VectorXd coord(3);
    coord[0] = ((ell % ((nx - 1) * (ny - 1))) % (nx - 1) + 1.) * hx;
    coord[1] = ((ell % ((nx - 1) * (ny - 1))) / (nx - 1) + 1) * hy;
    coord[2] = (ell / ((nx - 1) * (ny - 1)) + 1) * hz;
    return coord;
  }

  /** @brief get coarsened version of lattice */
  virtual std::shared_ptr<Lattice> get_coarse_lattice() const
  {
    if (not((nx % 2 == 0) and (ny % 2 == 0) and (nz % 2 == 0)))
    {
      std::cout << "ERROR: cannot coarsen lattice of size " << nx << " x " << ny << " x " << nz;
      std::cout << " [one of the extents is odd]" << std::endl;
      exit(-1);
    }
    if (not((nx / 2 > 1) and (ny / 2 > 1) and (nz / 2 > 1)))
    {
      std::cout << "ERROR: cannot coarsen lattice of size " << nx << " x " << ny << " x " << nz;
      std::cout << " [resulting lattice would have no interior vertices]" << std::endl;
      exit(-1);
    }
    return std::make_shared<Lattice3d>(nx / 2, ny / 2, nz / 2);
  }

  /** @brief get info string */
  virtual std::string get_info() const;

  /** @brief return lattice shape */
  inline virtual Eigen::VectorXi shape() const
  {
    Eigen::VectorXi s(3);
    s[0] = nx;
    s[1] = ny;
    s[2] = nz;
    return s;
  }

  /** @brief extent in x-direction */
  const unsigned int nx;
  /** @brief extent in y-direction */
  const unsigned int ny;
  /** @brief extent in z-direction */
  const unsigned int nz;
  /** @brief grid spacing in x-direction */
  const double hx;
  /** @brief grid spacing in y-direction */
  const double hy;
  /** @brief grid spacing in z-direction */
  const double hz;
};

#endif // LATTICE3D_HH
