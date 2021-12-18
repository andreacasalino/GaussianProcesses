/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>

namespace gauss::gp {
class CompositeKernelFunction : public KernelFunction {
public:
  CompositeKernelFunction(KernelFunctionPtr initial_element);

  void push_function(KernelFunctionPtr element);

  // evaluation should be reflexive: evaluate(a,b) = evaluate(b,a)
  double evaluate(const Eigen::VectorXd &a,
                          const Eigen::VectorXd &b) const override;

  std::unique_ptr<KernelFunction> copy() const override;

  std::vector<ParameterHandlerPtr> getParameters() const override;

protected:
  std::vector<KernelFunctionPtr> elements;
};
}
