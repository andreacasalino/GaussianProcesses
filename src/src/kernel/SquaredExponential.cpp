/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/Error.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

namespace gauss::gp {
SquaredExponential::SquaredExponential(const double teta0, const double teta1)
    : teta0(teta0), teta1(teta1) {
  teta0_squared = teta0 * teta0;
  teta1_squared = teta1 * teta1;
}

void SquaredExponential::setParameters(const std::vector<double> &values) {
  if (values.size() != 2) {
    throw Error{"Invalid parameters for SquaredExponential function"};
  }
  teta0 = values.front();
  teta1 = values.back();
  teta0_squared = teta0 * teta0;
  teta1_squared = teta1 * teta1;
}

namespace {
double squared_distance(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
  auto delta = a;
  delta -= b;
  return delta.squaredNorm();
}

double evaluate_exp_part(const double &squared_distance,
                         const double &teta1_squared) {
  return exp(-teta1_squared * squared_distance);
}
} // namespace

double SquaredExponential::evaluate(const Eigen::VectorXd &a,
                                    const Eigen::VectorXd &b) const {
  return teta0_squared *
         evaluate_exp_part(squared_distance(a, b), teta1_squared);
}

std::vector<double>
SquaredExponential::getGradient(const Eigen::VectorXd &a,
                                const Eigen::VectorXd &b) const {
  const auto squared_dist = squared_distance(a, b);
  const auto exp_part = evaluate_exp_part(squared_dist, teta1_squared);
  return {2.0 * teta0 * exp_part,
          -2.0 * teta1 * squared_dist * teta0_squared * exp_part};
}
} // namespace gauss::gp
