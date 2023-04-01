#ifndef LATTICE_HH
#define LATTICE_HH LATTICE_HH
#include <Eigen/Dense>

/** @file lattice2d.hh
 *
 * @brief two dimensional lattice
 */

/** @class Lattice
 *
 * @brief Abstract base class for lattices
 */
class Lattice
{
public:
  /** @brief Convert linear index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi idx_linear2euclidean(const unsigned int ell) const = 0;

  /** @brief Convert Euclidean index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int idx_euclidean2linear(const Eigen::VectorXi idx) const = 0;

  /** @brief Shift a linear index by an Euclideanvector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_index(const unsigned int ell, const Eigen::VectorXi shift) const = 0;
};

/** @class Lattice2d
 *
 * @brief Two dimensional structured lattice of size nx x ny
 */
class Lattice2d : public Lattice
{
public:
  /** @brief Create new instance
   *
   * @param[in] nx_ Extent in x-direction
   * @param[in] ny_ Extent in y-direction
   */
  Lattice2d(const unsigned int nx_, const unsigned int ny_) : nx(nx_), ny(ny_), M(nx_ * ny_) {}

  /** @brief Convert linear index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi idx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell < M);
    Eigen::VectorXi idx(2);
    idx[0] = ell % nx;
    idx[1] = ell / nx;
    return idx;
  }

  /** @brief Convert Euclidean index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int idx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    assert(idx[0] >= 0);
    assert(idx[0] < nx);
    assert(idx[1] >= 0);
    assert(idx[1] < ny);
    return ((idx[1] + ny) % ny) * nx + ((idx[0] + nx) % nx);
  };

  /** @brief Shift a linear index by an Euclideanvector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_index(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    int i = (ell % nx) + shift[0];
    int j = (ell / nx) + shift[1];
    return ((j + ny) % ny) * nx + ((i + nx) % nx);
  };

  /** @brief extent in x-direction */
  const unsigned int nx;
  /** @brief extent in y-direction */
  const unsigned int ny;
  /** @brief total number of lattice sites */
  const unsigned int M;
};

#endif // LATTICE_HH