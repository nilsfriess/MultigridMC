#ifndef LATTICE3D_HH
#define LATTICE3D_HH LATTICE3D_HH
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
 *  Points are arranged ordered lexicographically, for example for nx = 4, ny = 3, nz = 2
 *  (cell indices are projected onto the facets below):
 *
 * layer z=0
 *
 *  ^ y
 *  !
 *  !
 *
 * 15 ----- 16 ----- 17 ----- 18 ----- 19
 *  !        !        !        !        !
 *  !    8   !    9   !   10   !   11   !
 *  !        !        !        !        !                  N
 * 10 ----- 11 ----- 12 ----- 13 ----- 14              W  +  E
 *  !        !        !        !        !                  S
 *  !    4   !    5   !    6   !    7   !
 *  !        !        !        !        !
 *  5 ------ 6 ------ 7 ------ 8 ------ 9
 *  !        !        !        !        !
 *  !    0   !    1   !    2   !    3   !
 *  !        !        !        !        !
 *  0 ------ 1 ------ 2 ------ 3 ------ 4  ---> x
 *
 *  layer z=1
 *
 * 35 ----- 36 ----- 37 ----- 38 ----- 39
 *  !        !        !        !        !
 *  !   20   !   21   !   22   !   23   !
 *  !        !        !        !        !
 * 30 ----- 31 ----- 32 ----- 33 ----- 34
 *  !        !        !        !        !
 *  !   16   !   17   !   18   !   19   !
 *  !        !        !        !        !
 * 25 ----- 26 ----- 27 ----- 28 ----- 29
 *  !        !        !        !        !
 *  !   12   !   13   !   14   !   15   !
 *  !        !        !        !        !
 * 20 ----- 21 ----- 22 ----- 23 ----- 24
 *
 *  layer z=2
 *
 * 55 ----- 56 ----- 57 ----- 58 ----- 59
 *  !        !        !        !        !
 *  !        !        !        !        !
 *  !        !        !        !        !
 * 50 ----- 51 ----- 52 ----- 53 ----- 54
 *  !        !        !        !        !
 *  !        !        !        !        !
 *  !        !        !        !        !
 * 45 ----- 46 ----- 47 ----- 48 ----- 49
 *  !        !        !        !        !
 *  !        !        !        !        !
 *  !        !        !        !        !
 * 40 ----- 41 ----- 42 ----- 43 ----- 44
 *
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
                                      Lattice(nx_ * ny_ * nz_)
  {
    for (unsigned int ell = 0; ell < (nx + 1) * (ny + 1) * (nz + 1); ++ell)
    {
      Eigen::VectorXi idx = vertexidx_linear2euclidean(ell);
      if ((idx[0] == 0) or (idx[0] == nx) or (idx[1] == 0) or (idx[1] == ny) or (idx[2] == 0) or (idx[2] == nz))
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
    assert(ell < (nx + 1) * (ny + 1) * (nz + 1));
    Eigen::VectorXi idx(3);
    idx[0] = (ell % ((nx + 1) * (ny + 1))) % (nx + 1);
    idx[1] = (ell % ((nx + 1) * (ny + 1))) / (nx + 1);
    idx[2] = ell / ((nx + 1) * (ny + 1));
    return idx;
  }

  /** @brief Convert Euclidean vertex index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int vertexidx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    assert(idx[0] >= 0);
    assert(idx[0] < nx + 1);
    assert(idx[1] >= 0);
    assert(idx[1] < ny + 1);
    assert(idx[2] >= 0);
    assert(idx[2] < nz + 1);
    return idx[2] * (nx + 1) * (ny + 1) + idx[1] * (nx + 1) + idx[0];
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
  };

  /** @brief Shift a linear vertex index by an Euclidean vector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_vertexidx(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    assert(ell >= 0);
    assert(ell < (nx + 1) * (ny + 1) * (nz + 1));
    int i = (ell % ((nx + 1) * (ny + 1))) % (nx + 1) + shift[0];
    int j = (ell % ((nx + 1) * (ny + 1))) / (nx + 1) + shift[1];
    int k = ell / ((nx + 1) * (ny + 1)) + shift[2];
    assert(i >= 0);
    assert(i < nx + 1);
    assert(j >= 0);
    assert(j < ny + 1);
    assert(k >= 0);
    assert(k < nz + 1);
    return k * (nx + 1) * (ny + 1) + j * (nx + 1) + i;
  }

  /** @brief get equivalent index of vertex on next-finer lattice */
  virtual unsigned int fine_vertex_idx(const unsigned int ell) const
  {
    assert(ell < (nx + 1) * (ny + 1) * (nz + 1));
    int i = (ell % ((nx + 1) * (ny + 1))) % (nx + 1);
    int j = (ell % ((nx + 1) * (ny + 1))) / (nx + 1);
    int k = ell / ((nx + 1) * (ny + 1));
    return 2 * k * (2 * nx + 1) * (2 * ny + 1) + 2 * j * (2 * nx + 1) + 2 * i;
  }

  /** @brief get coarsened version of lattice */
  virtual std::shared_ptr<Lattice> get_coarse_lattice() const
  {
    assert(nx % 2 == 0);
    assert(ny % 2 == 0);
    assert(nz % 2 == 0);
    if (not((nx % 2 == 0) and (ny % 2 == 0) and (nz % 2 == 0)))
    {
      std::cout << "ERROR: cannot coarsen lattice of size " << nx << " x " << ny << " x " << nz << std::endl;
      exit(-1);
    }
    return std::make_shared<Lattice3d>(nx / 2, ny / 2, nz / 2);
  };

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
};

#endif // LATTICE3D_HH