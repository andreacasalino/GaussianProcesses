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
class CombinedFunction : public KernelFunction {
public:
  CombinedFunction(KernelFunctionPtr first_element,
                   KernelFunctionPtr second_element);

  /**
   * @brief Add an additional kernel function to the composite
   * @throw passing a null element
   */
  void add_element(KernelFunctionPtr element);

  double evaluate(const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) const override;

  std::vector<ParameterCnstPtr> getParameters() const override;

protected:
  std::vector<KernelFunctionPtr> elements;
};
} // namespace gauss::gp
