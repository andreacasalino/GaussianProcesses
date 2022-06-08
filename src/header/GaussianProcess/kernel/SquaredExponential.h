/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/RadialFunction.h>

namespace gauss::gp {
class RadialExponential : public RadialFunction {
public:
  RadialExponential(double teta0, double teta1);

  RadialFunctionPtr copy() const final;

  double evaluate(const double squared_distance) const final;

  std::vector<double> getGradient(const double squared_distance) const final;

private:
  double teta0_squared;
  double teta1_squared;
};

/**
 * @brief Kernel function k(x1, x2) assumed equal to:
 * teta0^2 * exp(-teta1^2 * (x1-x2).dot(x1-x2))
 */
class SquaredExponential : public RadialKernelFunction {
public:
  SquaredExponential(const double teta0, const double teta1);
};
} // namespace gauss::gp
