#ifndef LATTICE_HH
#define LATTICE_HH LATTICE_HH
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
   * @param[in] M_ number of lattice sites
   */
  Lattice(const unsigned int M_) : M(M_) {}

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

  /** @brief return lattice shape */
  inline virtual Eigen::VectorXi shape() const = 0;

  /** @brief return lattice dimension */
  inline virtual int dim() const { return shape().size(); }

  /** @brief get coarsened version of lattice */
  virtual std::shared_ptr<Lattice> get_coarse_lattice() const = 0;

  /** @brief get info string */
  virtual std::string get_info() const = 0;

  /** @brief total number of lattice sites */
  const unsigned int M;
};

#endif // LATTICE_HH