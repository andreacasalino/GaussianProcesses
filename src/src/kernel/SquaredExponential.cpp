/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/kernel/SquaredExponential.h>

#include <math.h>

namespace gauss::gp {
RadialExponential::RadialExponential(const double sigma, const double length)
    : RadialFunction(std::vector<double>{sigma, length}) {
  sigma_squared = sigma * sigma;
  length_squared = length * length;
}

RadialFunctionPtr RadialExponential::copy() const {
  const auto &params = getParameters();
  return std::make_unique<RadialExponential>(params.front(), params.back());
}

namespace {
double evaluate_exp_part(const double &squared_distance,
                         const double &length_squared) {
  return exp(-squared_distance / length_squared);
}
} // namespace

double RadialExponential::evaluate(const double squared_distance) const {
  return sigma_squared * evaluate_exp_part(squared_distance, length_squared);
}

std::vector<double>
RadialExponential::getGradient(const double squared_distance) const {
  const auto exp_part = evaluate_exp_part(squared_distance, length_squared);
  const auto &params = getParameters();
  const auto &sigma = params.front();
  const auto &length = params.back();
  return {2.0 * sigma * exp_part,
          sigma_squared * exp_part * 2.0 * squared_distance / pow(length, 3)};
}

SquaredExponential::SquaredExponential(const double sigma, const double length)
    : RadialKernelFunction(std::make_unique<RadialExponential>(sigma, length)) {
}

} // namespace gauss::gp
