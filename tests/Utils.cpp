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
} // namespace gauss::gp::test
