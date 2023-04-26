#ifndef STATISTICS_HH
#define STATISTICS_HH STATISTICS_HH

/** @file statistics.hh
 * @brief Header file for statistics class
 */

#include <cmath>
#include <deque>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <Eigen/Dense>

/** @class Statistics
 * @brief Class for recording statistics of an observable
 *
 * This class can be used to collect statistics on a random observable
 * \f$Q\f$, which is assumed to be a n--dimensional vector.
 * The class allows calculation of the following quantities:
 *
 * - *Estimated average*
 *   \f[
 *     \overline{Q} = \langle Q_i \rangle = \frac{1}{N}\sum_{i=0}^{N-1} Q_i
 *   \f]
 *
 * - *Estimated covariance matrix*
 *   \f[
 *     \text{Var}[Q] = \frac{1}{N-1}\sum_{i=0}^{N-1}(Q_i-\overline{Q}) (Q_i-\overline{Q})^T
 *   \f]
 *
 * - *Auto covariance function*
 *   \f[
 *     C(k) = \langle (Q_i-\overline{Q})(Q_{i+k}-\overline{Q})^T \rangle\\
 *          = \langle Q_i Q_{i+k}^T \rangle - \overline{Q}^2
 *   \f]
 *
 * - *Integrated auto covariance matrix*
 *   \f[
 *     \tau_{\text{int}} = Id + 2 \sum_{k=1}^{N-1}\left(1-\frac{k}{N}\right)
 *                         C(k) C(0)^{-1}
 *   \f]
 *
 * To achieve this, the calculates the quantities
 * \f[
 *    S_k = \frac{1}{N_k} \sum_{i=k}^{N-1} Q_i Q_{i+k}^T
 * \f]
 * for \f$k=0,\dots,k_{\max}\f$ where \f$N_k:=N-k\f$.
 *
 */
class Statistics
{
public:
  /** @brief Create a new instance
   *
   * @param[in] label_ Label for identifying object
   * @param[in] autocorr_window_ Window over which autocorrelations are measured.
   */
  Statistics(const std::string label_,
             const unsigned int autocorr_window_) : label(label_),
                                                    k_max(autocorr_window_)
  {
    reset();
  }

  /** @brief Return label */
  std::string get_label() const { return label; }

  /** @brief Reset all counters */
  void reset()
  {
    Q_k.clear();
    S_k.clear();
    n_samples = 0;
  }

  /** @brief Record a new sample
   *
   * @param[in] Q Value of new sample
   */
  void record_sample(const Eigen::VectorXd Q);

  /** @brief Return estimator for covariance matrix
   */
  Eigen::MatrixXd covariance() const;

  /** @brief Return estimator for average */
  Eigen::VectorXd average() const;

  /** @brief Return vector with auto covariance function \f$C(k)\f$ */
  std::vector<Eigen::MatrixXd> auto_covariance() const;

  /** @brief Return integrated auto covariance time \f$\tau_{\text{int}}\f$ */
  Eigen::MatrixXd tau_int() const;

  /** @brief Return size of autocorrelation window */
  unsigned int autocorr_window() const { return k_max; }

  /** @brief Return the number of samples samples */
  unsigned int samples() const;

private:
  /** @brief Label for identifying object */
  const std::string label;
  /** @brief Window over which auto-correlations are measured */
  const unsigned int k_max;
  /** @brief Number of collected samples */
  unsigned int n_samples;
  /** @brief Deque holding the last samples
   * This is necessary to update autocorrelations. The deque stores (in this
   * order) \f$Q_j, Q_{j-1}, \dots, Q_{j-k_{\max}}\f$.
   */
  std::deque<Eigen::VectorXd> Q_k;
  /** @brief Vector with estimated autocorrelations
   * This stores (in this order) \f$S_0,S_1,\dots,S_{k_{\max}}\f$.
   */
  std::vector<Eigen::MatrixXd> S_k;
  /** @brief Running average for quantity */
  Eigen::VectorXd avg;
  /** @brief Running average for Q^2 */
  Eigen::MatrixXd avg2;
};

std::ostream &operator<<(std::ostream &os, const Statistics &stats);

#endif // STATISTICS_HH
