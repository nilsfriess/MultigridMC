#ifndef LATTICE2D_HH
#define LATTICE2D_HH LATTICE2D_HH
/** @file lattice2d.hh
 *
 * @brief two dimensional lattice
 */

/** @class Lattice2d
 *
 * @brief Two dimensional structured lattice of size nx x ny
 */
class Lattice2d
{
public:
  /** @brief Create new instance
   *
   * @param[in] nx_ Extent in x-direction
   * @param[in] ny_ Extent in y-direction
   */
  Lattice2d(const unsigned int nx_, const unsigned int ny_) : nx(nx_), ny(ny_), M(nx_ * ny_) {}

  /** @brief extent in x-direction */
  const unsigned int nx;
  /** @brief extent in y-direction */
  const unsigned int ny;
  /** @brief total number of lattice sites */
  const unsigned int M;
};

#endif // LATTICE2D_HH