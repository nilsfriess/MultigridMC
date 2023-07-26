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
 *     0       1       2       3      (cells)
 * 0 ----- 1 ----- 2 ----- 3 ----- 4  (vertices)
 *
 */
class Lattice1d : public Lattice
{
public:
  /** @brief Create new instance
   *
   * @param[in] n_ Extent of lattice (= number of cells)
   */
  Lattice1d(const unsigned int n_) : n(n_), Lattice(n_)
  {
    for (int ell = 0; ell < n + 1; ++ell)
    {
      if ((ell == 0) or (ell == n))
      {
        boundary_vertex_idxs->push_back(ell);
      }
      else
      {
        interior_vertex_idxs->push_back(ell);
      }
    }
  }

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

  /** @brief Convert Euclidean index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int cellidx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    assert(idx[0] >= 0);
    assert(idx[0] < n);
    return idx[0];
  };

  /** @brief Convert linear vertex index to Euclidean index
   *
   * @param[in] ell linear index to be converted
   */
  inline virtual Eigen::VectorXi vertexidx_linear2euclidean(const unsigned int ell) const
  {
    assert(ell >= 0);
    assert(ell < Ncell + 1);
    Eigen::VectorXi idx(1);
    idx[0] = ell;
    return idx;
  }

  /** @brief Convert Euclidean vertex index to linear index
   *
   * @param[in] idx Euclidean index to be converted
   */
  inline virtual unsigned int vertexidx_euclidean2linear(const Eigen::VectorXi idx) const
  {
    assert(idx[0] >= 0);
    assert(idx[0] < n + 1);
    return idx[0];
  };

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
  };

  /** @brief Shift a linear vertex index by an Euclidean vector
   *
   * @param[in] idx Euclidean index to be shifted
   * @param[in] shift Euclidean shift vector
   */
  inline virtual unsigned int shift_vertexidx(const unsigned int ell, const Eigen::VectorXi shift) const
  {
    int i = ell + shift[0];
    assert(i >= 0);
    assert(i < n + 1);
    return i;
  };

  /** @brief get equivalent index of vertex on next-finer lattice */
  virtual unsigned int fine_vertex_idx(const unsigned int ell) const
  {
    return 2 * ell;
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