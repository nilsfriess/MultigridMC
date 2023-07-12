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
 * @brief One dimensional structured lattice of size n
 *
 * Periodic boundary conditions are implicitly assumed
 */
class Lattice1d : public Lattice
{
public:
  /** @brief Create new instance
   *
   * @param[in] n_ Extent of lattice (= number of sites)
   */
  Lattice1d(const unsigned int n_) : n(n_), Lattice(n_) {}

  /** @brief Convert linear index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi idx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell < M);
    Eigen::VectorXi idx(1);
    idx[0] = ell;
    return idx;
  }

  /** @brief Convert Euclidean index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int idx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    return (idx[0] + n) % n;
  };

  /** @brief Shift a linear index by an Euclideanvector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_index(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    int i = (ell % n) + shift[0];
    return ((i + n) % n);
  };

  /** @brief get coarsened version of lattice */
  virtual std::shared_ptr<Lattice> get_coarse_lattice() const
  {
    assert(n % 2 == 0);
    return std::make_shared<Lattice1d>(n / 2);
  };

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