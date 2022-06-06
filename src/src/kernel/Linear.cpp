/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/Error.h>
#include <GaussianProcess/kernel/Linear.h>

namespace gauss::gp {
LinearFunction::LinearFunction(const double teta0, const double teta1,
                               const std::size_t space_size)
    : teta0(teta0), teta1(teta1), mean(static_cast<Eigen::Index>(space_size)) {
  if (0 == space_size) {
    throw Error{"Invalid mean size"};
  }
  mean.setZero();
  teta0_squared = teta0 * teta0;
  teta1_squared = teta1 * teta1;
}

LinearFunction::LinearFunction(const double teta0, const double teta1,
                               const Eigen::VectorXd &mean)
    : LinearFunction{teta0, teta1, static_cast<std::size_t>(mean.size())} {
  this->mean = mean;
}

std::vector<double> LinearFunction::getParameters() const {
  std::vector<double> result = {teta0, teta1};
  for (const auto &val : mean) {
    result.push_back(val);
  }
  return result;
}

void LinearFunction::setParameters(const std::vector<double> &values) {
  if (values.size() != numberOfParameters()) {
    throw Error{"Invalid parameters for LinearFunction function"};
  }
  teta0 = values[0];
  teta1 = values[1];
  teta0_squared = teta0 * teta0;
  teta1_squared = teta1 * teta1;
  std::size_t k = 2;
  for (auto &val : mean) {
    val = values[k];
    ++k;
  }
}

namespace {
double dot_mean(const Eigen::VectorXd &a, const Eigen::VectorXd &b,
                const Eigen::VectorXd &mean) {
  return (a - mean).dot(b - mean);
}
} // namespace

double LinearFunction::evaluate(const Eigen::VectorXd &a,
                                const Eigen::VectorXd &b) const {
  return teta0_squared * teta1_squared * dot_mean(a, b, mean);
}

std::vector<double>
LinearFunction::getGradient(const Eigen::VectorXd &a,
                            const Eigen::VectorXd &b) const {

  std::vector<double> result;
  result.push_back(2.0 * teta0);
  result.push_back(2.0 * teta1 * dot_mean(a, b, mean));
  const auto result_back = teta1_squared * (2.0 * mean - a - b);
  for (const auto &val : result_back) {
    result.push_back(val);
  }
  return result;
}
} // namespace gauss::gp
