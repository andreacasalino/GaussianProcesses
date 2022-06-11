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
    : RadialFunction(std::vector<double>{sigma, length}) {}

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
  const auto &sigma = getParameters().front();
  const auto &length = getParameters().back();
  return sigma * sigma * evaluate_exp_part(squared_distance, length * length);
}

std::vector<double>
RadialExponential::getGradient(const double squared_distance) const {
  const auto &sigma = getParameters().front();
  const auto &length = getParameters().back();
  const auto exp_part = evaluate_exp_part(squared_distance, length * length);
  const auto &params = getParameters();
  return {2.0 * sigma * exp_part,
          sigma * sigma * exp_part * 2.0 * squared_distance / pow(length, 3)};
}

SquaredExponential::SquaredExponential(const double sigma, const double length)
    : RadialKernelFunction(std::make_unique<RadialExponential>(sigma, length)) {
}

} // namespace gauss::gp
