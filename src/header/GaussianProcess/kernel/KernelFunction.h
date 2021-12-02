/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <Eigen/Core>
#include <memory>

namespace gauss::gp {
class KernelFunction {
public:
  virtual ~KernelFunction() = default;

  virtual double evaluate(const Eigen::VectorXd &a,
                          const Eigen::VectorXd &b) const = 0;

  virtual std::unique_ptr<KernelFunction> copy() const = 0;

protected:
  KernelFunction() = default;
};
using KernelFunctionPtr = std::unique_ptr<KernelFunction>;
} // namespace gauss::gp
