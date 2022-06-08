/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/kernel/SquaredExponential.h>

#include <math.h>

namespace gauss::gp {
RadialExponential::RadialExponential(const double teta0, const double teta1)
    : RadialFunction(std::vector<double>{teta0, teta1}) {
  teta0_squared = teta0 * teta0;
  teta1_squared = teta1 * teta1;
}

RadialFunctionPtr RadialExponential::copy() const {
  const auto &params = getParameters();
  return std::make_unique<RadialExponential>(params.front(), params.back());
}

namespace {
double evaluate_exp_part(const double &squared_distance,
                         const double &teta1_squared) {
  return exp(-teta1_squared * squared_distance);
}
} // namespace

double RadialExponential::evaluate(const double squared_distance) const {
  return teta0_squared * evaluate_exp_part(squared_distance, teta1_squared);
}

std::vector<double>
RadialExponential::getGradient(const double squared_distance) const {
  const auto exp_part = evaluate_exp_part(squared_distance, teta1_squared);
  const auto &params = getParameters();
  const auto &teta0 = params.front();
  const auto &teta1 = params.back();
  return {2.0 * teta0 * exp_part,
          -2.0 * teta1 * squared_distance * teta0_squared * exp_part};
}

SquaredExponential::SquaredExponential(const double teta0, const double teta1)
    : RadialKernelFunction(std::make_unique<RadialExponential>(teta0, teta1)) {}

} // namespace gauss::gp
