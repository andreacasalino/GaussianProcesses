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
  CombinedFunction(KernelFunctionPtr first, KernelFunctionPtr second);

  std::size_t numberOfParameters() const final { return params_numb; }

  std::vector<double> getParameters() const final;

  void setParameters(const std::vector<double> &values) const final;

  double evaluate(const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) const final;

  std::vector<double> get_gradient(const Eigen::VectorXd &a,
                                   const Eigen::VectorXd &b) const final;

  void addElement(KernelFunctionPtr element);
  const std::vector<KernelFunctionPtr> &getElements() const {
    return elements;
  };

private:
  std::vector<KernelFunctionPtr> elements;
  std::size_t params_numb = 0;
};
} // namespace gauss::gp
