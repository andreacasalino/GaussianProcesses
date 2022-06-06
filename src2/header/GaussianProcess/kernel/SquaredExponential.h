/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>

namespace gauss::gp {
/**
 * @brief Kernel function k(x1, x2) assumed equal to:
 * teta0^2 * exp(-teta1^2 * (x1-x2).dot(x1-x2))
 */
class SquaredExponential : public KernelFunction {
public:
  SquaredExponential(const double teta0, const double teta1);

  std::unique_ptr<KernelFunction> copy() const override {
    return std::make_unique<SquaredExponential>(teta0, teta1);
  }

  std::size_t numberOfParameters() const override { return 2; }

  std::vector<double> getParameters() const override { return {teta0, teta1}; }

  void setParameters(const std::vector<double> &values) override;

  double evaluate(const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) const override;

  std::vector<double> getGradient(const Eigen::VectorXd &a,
                                  const Eigen::VectorXd &b) const override;

private:
  double teta0;
  double teta0_squared;
  double teta1;
  double teta1_squared;
};
} // namespace gauss::gp
