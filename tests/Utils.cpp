/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Utils.h"

#include <GaussianProcess/Error.h>

namespace gauss::gp::test {
std::vector<Eigen::VectorXd> make_samples(const std::size_t samples_numb,
                                          const Eigen::Index sample_size) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples_numb);
  for (std::size_t k = 0; k < samples_numb; ++k) {
    result.emplace_back(sample_size).setRandom();
  }
  return result;
}

bool is_zeros(const Eigen::MatrixXd &subject, const double toll) {
  for (Eigen::Index r = 0; r < subject.rows(); ++r) {
    for (Eigen::Index c = 0; c < subject.cols(); ++c) {
      if (std::abs(subject(r, c)) > toll) {
        return false;
      }
    }
  }
  return true;
}

bool is_equal(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b,
              const double toll) {
  return is_zeros(a - b, toll);
}

bool is_equal_vec(const Eigen::VectorXd &a, const Eigen::VectorXd &b,
                  const double toll) {
  if (a.size() != b.size()) {
    return false;
  }
  for (Eigen::Index k = 0; k < a.size(); ++k) {
    if (std::abs(a(k) - b(k)) > toll) {
      return false;
    }
  }
  return true;
}

bool is_symmetric(const Eigen::MatrixXd &subject, const double toll) {
  return is_equal(subject, subject.transpose(), toll);
}

bool is_inverse(const Eigen::MatrixXd &subject,
                const Eigen::MatrixXd &candidate, const double toll) {
  return is_equal(subject * candidate,
                  Eigen::MatrixXd::Identity(subject.rows(), subject.cols()),
                  toll);
}
} // namespace gauss::gp::test
