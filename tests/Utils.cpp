/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Utils.h"

#include <GaussianProcess/Error.h>

namespace gauss::gp::test {
TestFunction::TestFunction(const double teta) : teta(teta) {}

void TestFunction::setParameters(const std::vector<double> &values) {
  if (values.size() != 1) {
    throw Error{"Invalid values"};
  }
}

double TestFunction::evaluate(const Eigen::VectorXd &a,
                              const Eigen::VectorXd &b) const {
  return teta * a.dot(b);
}

std::vector<double> TestFunction::getGradient(const Eigen::VectorXd &a,
                                              const Eigen::VectorXd &b) const {
  return {a.dot(b)};
}

std::vector<Eigen::VectorXd> make_samples(const std::size_t samples_numb,
                                          const Eigen::Index sample_size,
                                          const double delta) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples_numb);
  double val = 0;
  for (std::size_t k = 0; k < samples_numb; ++k, val += delta) {
    auto &new_sample = result.emplace_back(sample_size).setOnes();
    new_sample *= val;
  }
  return result;
}
} // namespace gauss::gp::test
