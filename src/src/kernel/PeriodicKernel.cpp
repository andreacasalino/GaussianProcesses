/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/kernel/PeriodicFunction.h>

namespace gauss::gp {
RadialPeriodic::RadialPeriodic(const double sigma, const double length,
                               const double period)
    : RadialFunction(std::vector<double>{sigma, length, period}) {}

RadialFunctionPtr RadialPeriodic::copy() const {
  const auto &params = getParameters();
  return std::make_unique<RadialPeriodic>(params.front(), params[1],
                                          params.back());
}

namespace {
static constexpr double PI_2 = 2.0 * 3.14159265359;

double get_angle(const double &squared_distance, const double &period) {
  return PI_2 * sqrt(squared_distance) / period;
}
} // namespace

double RadialPeriodic::evaluate(const double squared_distance) const {
  const auto &params = getParameters();
  const auto &sigma = params.front();
  const auto &length = params[1];
  const auto &period = params.back();
  const auto sin_angle = sin(get_angle(squared_distance, period));
  return sigma * sigma * exp(-sin_angle * sin_angle / length * length);
}

std::vector<double>
RadialPeriodic::getGradient(const double squared_distance) const {
  const auto &params = getParameters();
  const auto &sigma = params.front();
  const auto &length = params[1];
  const auto &period = params.back();
  const auto angle = get_angle(squared_distance, period);
  const auto length_squared = length * length;
  const auto sin_angle = sin(angle);
  const auto exp_arg = -sin_angle * sin_angle / length_squared;
  const auto exp_part = exp(exp_arg);
  const auto sigma_squared = sigma * sigma;
  return {2.0 * sigma * exp_part,
          sigma_squared * exp_part * 2.0 * (-exp_arg) / length,
          sigma_squared * exp_part * 2.0 * sin_angle * cos(angle) * angle /
              (length_squared * period)};
}

PeriodicFunction::PeriodicFunction(const double sigma, const double length,
                                   const double period)
    : RadialKernelFunction(
          std::make_unique<RadialPeriodic>(sigma, length, period)) {}
} // namespace gauss::gp
