#ifndef LATTICE2D_HH
#define LATTICE2D_HH LATTICE2D_HH
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
 * 15 ----- 16 ----- 17 ----- 18 ----- 19
 *  !        !        !        !        !
 *  !    8   !    9   !   10   !   11   !
 *  !        !        !        !        !                 N
 * 10 ----- 11 ----- 12 ----- 13 ----- 14              W  +  E
 *  !        !        !        !        !                 S
 *  !    4   !    5   !    6   !    7   !
 *  !        !        !        !        !
 *  5 ------ 6 ------ 7 ------ 8 ------ 9
 *  !        !        !        !        !
 *  !    0   !    1   !    2   !    3   !
 *  !        !        !        !        !
 *  0 ------ 1 ------ 2 ------ 3 ------ 4  ---> x
 *
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
                                                              Lattice(nx_ * ny_)
  {
    for (unsigned int ell = 0; ell < (nx + 1) * (ny + 1); ++ell)
    {
      Eigen::VectorXi idx = vertexidx_linear2euclidean(ell);
      if ((idx[0] == 0) or (idx[0] == nx) or (idx[1] == 0) or (idx[1] == ny))
      {
        boundary_vertex_idxs->push_back(ell);
      }
      else
      {
        interior_vertex_idxs->push_back(ell);
      }
    }
  }

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
    assert(ell < (nx + 1) * (ny + 1));
    Eigen::VectorXi idx(2);
    idx[0] = ell % (nx + 1);
    idx[1] = ell / (nx + 1);
    return idx;
  }

  /** @brief Convert Euclidean vertex index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int vertexidx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    assert(idx[0] >= 0);
    assert(idx[0] < nx);
    assert(idx[1] >= 0);
    assert(idx[1] < ny);
    return idx[1] * (nx + 1) + idx[0];
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
  };

  /** @brief Shift a linear vertex index by an Euclidean vector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_vertexidx(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    assert(ell < (nx + 1) * (ny + 1));
    int i = (ell % (nx + 1)) + shift[0];
    int j = (ell / (nx + 1)) + shift[1];
    return j * (nx + 1) + i;
  }

  /** @brief get equivalent index of vertex on next-finer lattice */
  virtual unsigned int fine_vertex_idx(const unsigned int ell) const
  {
    int i = (ell % (nx + 1));
    int j = (ell / (nx + 1));
    assert(i >= 0);
    assert(i < nx + 1);
    assert(j >= 0);
    assert(j < ny + 1);
    return 2 * j * (2 * nx + 1) + 2 * i;
  }

  /** @brief get coarsened version of lattice */
  virtual std::shared_ptr<Lattice> get_coarse_lattice() const
  {
    assert(nx % 2 == 0);
    assert(ny % 2 == 0);
    if (not((nx % 2 == 0) and (ny % 2 == 0)))
    {
      std::cout << "ERROR: cannot coarsen lattice of size " << nx << " x " << ny << std::endl;
      exit(-1);
    }
    return std::make_shared<Lattice2d>(nx / 2, ny / 2);
  };

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
};

#endif // LATTICE2D_HH