/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/ParameterHandler.h>

namespace gauss::gp {
class KernelFunction {
public:
  virtual ~KernelFunction() = default;

  // evaluation should be reflexive: evaluate(a,b) = evaluate(b,a)
  virtual double evaluate(const Eigen::VectorXd &a,
                          const Eigen::VectorXd &b) const = 0;

  virtual std::unique_ptr<KernelFunction> copy() const = 0;

  virtual std::vector<ParameterHandlerPtr> getParameters() const = 0;

protected:
  KernelFunction() = default;
};
using KernelFunctionPtr = std::unique_ptr<KernelFunction>;
} // namespace gauss::gp
