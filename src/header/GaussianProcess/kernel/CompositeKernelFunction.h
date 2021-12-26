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
 * @brief The composite sums all the individual wrapped kernel functions.
 *
 */
class CompositeKernelFunction : public KernelFunction {
public:
  CompositeKernelFunction(KernelFunctionPtr initial_element);

  /**
   * @brief Add an additional kernel function to the composite
   * @throw passing a null element
   */
  void push_function(KernelFunctionPtr element);

  double evaluate(const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) const override;

  std::unique_ptr<KernelFunction> copy() const override;

  std::vector<ParameterHandlerPtr> getParameters() const override;

protected:
  std::vector<KernelFunctionPtr> elements;
};
} // namespace gauss::gp
