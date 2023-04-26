#include "statistics.hh"

/* Record a new sample */
void Statistics::record_sample(const Eigen::VectorXd Q)
{
  n_samples++;
  if (n_samples == 1)
  {
    // First samples
    avg = Q;
    avg2 = Q * Q.transpose();
  }
  else
  {
    // Update running averages
    avg += (Q - avg) / (1.0 * n_samples);
    avg2 += (Q * Q.transpose() - avg2) / (1.0 * n_samples);
  }
  Q_k.push_front(Q);
  // Renove last sample from deque
  if (Q_k.size() > k_max)
  {
    Q_k.pop_back();
  }

  // Update running S_k
  for (unsigned int k = 0; k < Q_k.size(); ++k)
  {
    unsigned int N_k = n_samples - k;
    if (N_k == 1)
    {
      S_k.push_back(Q_k[0] * Q_k[k].transpose());
    }
    else
    {
      S_k[k] += (Q_k[0] * Q_k[k].transpose() - S_k[k]) / (1.0 * N_k);
    }
  }
}

/* Return estimator for co variance */
Eigen::MatrixXd Statistics::covariance() const
{
  return 1.0 * n_samples / (n_samples - 1.0) * (avg2 - avg * avg.transpose());
}

/* Return estimator for average */
Eigen::VectorXd Statistics::average() const
{
  return avg;
}

/* Return vector with autocovariance function \f$C(k)\f$ */
std::vector<Eigen::MatrixXd> Statistics::auto_covariance() const
{
  std::vector<Eigen::MatrixXd> autocov;
  for (auto it = S_k.begin(); it != S_k.end(); ++it)
  {
    autocov.push_back(*it - avg * avg.transpose());
  }
  return autocov;
}

/* Return integrated auto covariance matrix \f$\tau_{\text{int}}\f$ */
Eigen::MatrixXd Statistics::tau_int() const
{
  const std::vector<Eigen::MatrixXd> &C_k_ = auto_covariance();
  Eigen::MatrixXd covariance = C_k_[0];
  unsigned int dim = covariance.rows();
  Eigen::MatrixXd tau_int_tmp = covariance;
  unsigned int kmax = C_k_.size();
  for (unsigned int k = 1; k < kmax; ++k)
  {
    tau_int_tmp += 2 * (1. - k / (1.0 * kmax)) * C_k_[k];
  }
  return tau_int_tmp * covariance.inverse();
}

/* Return the number of samples (across all processors) */
unsigned int Statistics::samples() const
{
  return n_samples;
}

/* Output statistics to stream object */
std::ostream &operator<<(std::ostream &os, const Statistics &stats)
{
  os << " ";
  os << std::setprecision(6) << std::fixed;
  os << stats.get_label() << ": Avg = " << stats.average() << std::endl;
  os << " " << stats.get_label() << ": Var = " << stats.covariance() << std::endl;
  os << std::setprecision(3) << std::fixed;
  os << " " << stats.get_label() << ": tau_{int}   = " << stats.tau_int()
     << std::endl;
  os << " " << stats.get_label() << ": window      = " << stats.autocorr_window()
     << std::endl;
  os << " " << stats.get_label() << ": # samples   = " << stats.samples()
     << std::endl;
  return os;
}
