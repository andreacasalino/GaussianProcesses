/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>
#include <GaussianProcess/kernel/ParameterHandler.h>

namespace gauss::gp {
/**
 * @brief Kernel function k(x1, x2) assumed equal to:
 * teta0^2 * exp(-teta1^2 * (x1-x2).dot(x1-x2))
 */
class SquaredExponential : public KernelFunction {
public:
  SquaredExponential(const double teta0, const double teta1);

  double evaluate(const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) const override;

  std::unique_ptr<KernelFunction> copy() const override;

  std::vector<ParameterHandlerPtr> getParameters() const override;

private:
  Parameter teta0;
  Parameter teta1;
};
} // namespace gauss::gp
