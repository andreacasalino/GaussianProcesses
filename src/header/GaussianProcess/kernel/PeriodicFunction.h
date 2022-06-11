/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/RadialFunction.h>

namespace gauss::gp {
/**
 * @brief refer to
 * https://peterroelants.github.io/posts/gaussian-process-kernels/#Periodic-kernel
 *
 */
class RadialPeriodic : public RadialFunction {
public:
  RadialPeriodic(const double sigma, const double length, const double period);

  RadialFunctionPtr copy() const final;

  double evaluate(const double squared_distance) const final;

  std::vector<double> getGradient(const double squared_distance) const final;
};

/**
 * @brief
 * https://peterroelants.github.io/posts/gaussian-process-kernels/#Periodic-kernel
 */
class PeriodicFunction : public RadialKernelFunction {
public:
  PeriodicFunction(const double sigma, const double length,
                   const double period);
};
} // namespace gauss::gp
