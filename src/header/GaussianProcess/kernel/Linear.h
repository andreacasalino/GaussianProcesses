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
 * teta0^2 + teta1^2 * (x1 - mean).dot( (x2 -mean) )
 */
class LinearFunction : public KernelFunction {
public:
  LinearFunction(const double teta0, const double teta1,
                 const std::size_t space_size);

  LinearFunction(const double teta0, const double teta1,
                 const Eigen::VectorXd &mean);

  std::unique_ptr<KernelFunction> copy() const override {
    return std::make_unique<LinearFunction>(teta0, teta1, mean);
  }

  std::size_t numberOfParameters() const override {
    return 2 + static_cast<std::size_t>(mean.size());
  }

  std::vector<double> getParameters() const override;

  void setParameters(const std::vector<double> &values) override;

  double evaluate(const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) const override;

  std::vector<double> getGradient(const Eigen::VectorXd &a,
                                  const Eigen::VectorXd &b) const override;

private:
  double teta0;
  double teta1;
  Eigen::VectorXd mean;
};
} // namespace gauss::gp
