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
 * @brief Three dimensional structured lattice of size nx x ny x nz with periodic boundary
 *
 * Periodic boundary conditions are implicitly assumed
 *
 *  Points are arranged ordered lexicographically, for example for nx = 4, ny = 3, nz = 2:
 *
 *  z = 0                                          z = 1
 *
 *  ^ y                                            ^ y
 *  !                                              !
 *                                    N
 *  8 ---- 9 --- 10 --- 11         W  +  E        20 --- 21 --- 22 --- 23
 *  !      !      !      !            S            !      !      !      !
 *  !      !      !      !                         !      !      !      !
 *  4 ---- 5 ---- 6 ---- 7                        16 --- 17 --- 18 --- 19
 *  !      !      !      !                         !      !      !      !
 *  !      !      !      !                         !      !      !      !
 *  0 ---- 1 ---- 2 ---- 3  ---> x                12 --- 13 --- 14 --- 15  ---> x
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
            const unsigned int nz_) : nx(nx_), ny(ny_), nz(nz_), Lattice(nx_ * ny_ * nz_) {}

  /** @brief Convert linear index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi idx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell < M);
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
  inline virtual unsigned int idx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    return ((idx[2] + nz) % nz) * nx * ny + ((idx[1] + ny) % ny) * nx + ((idx[0] + nx) % nx);
  };

  /** @brief Shift a linear index by an Euclideanvector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_index(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    int i = (ell % (nx * ny)) % nx + shift[0];
    int j = (ell % (nx * ny)) / nx + shift[1];
    int k = ell / (nx * ny) + shift[2];
    return ((k + nz) % nz) * nx * ny + ((j + ny) % ny) * nx + ((i + nx) % nx);
  };

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