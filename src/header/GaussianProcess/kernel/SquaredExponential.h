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
 * https://peterroelants.github.io/posts/gaussian-process-kernels/#Exponentiated-quadratic-kernel
 *
 */
class RadialExponential : public RadialFunction {
public:
  RadialExponential(const double sigma, const double length);

  RadialFunctionPtr copy() const final;

  double evaluate(const double squared_distance) const final;

  std::vector<double> getGradient(const double squared_distance) const final;

private:
  double sigma_squared;
  double length_squared;
};

/**
 * @brief
 * https://peterroelants.github.io/posts/gaussian-process-kernels/#Exponentiated-quadratic-kernel
 */
class SquaredExponential : public RadialKernelFunction {
public:
  SquaredExponential(const double sigma, const double length);
};
} // namespace gauss::gp
