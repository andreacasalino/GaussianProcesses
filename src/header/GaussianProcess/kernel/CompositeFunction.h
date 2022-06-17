/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>

namespace gauss::gp {
class CompositeFunction : public KernelFunction {
public:
  std::size_t numberOfParameters() const final { return params_numb; }

  std::vector<double> getParameters() const final;

  void setParameters(const std::vector<double> &values) final;

  void addElement(KernelFunctionPtr element);
  const std::vector<KernelFunctionPtr> &getElements() const {
    return elements;
  };

protected:
  CompositeFunction(KernelFunctionPtr first, KernelFunctionPtr second);
  CompositeFunction(const CompositeFunction &o);

private:
  std::vector<KernelFunctionPtr> elements;
  std::size_t params_numb = 0;
};

/**
 * @brief The composite sums all the individual wrapped kernel functions.
 *
 */
class Summation : public CompositeFunction {
public:
  Summation(KernelFunctionPtr first, KernelFunctionPtr second);

  KernelFunctionPtr copy() const final;

  double evaluate(const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) const final;

  std::vector<double> getGradient(const Eigen::VectorXd &a,
                                  const Eigen::VectorXd &b) const final;

private:
  Summation(const Summation &o);
};

/**
 * @brief The composite make a product product of all the individual wrapped
 * kernel functions.
 *
 */
class Product : public CompositeFunction {
public:
  Product(KernelFunctionPtr first, KernelFunctionPtr second);

  KernelFunctionPtr copy() const final;

  double evaluate(const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) const final;

  std::vector<double> getGradient(const Eigen::VectorXd &a,
                                  const Eigen::VectorXd &b) const final;

private:
  Product(const Product &o);
};
} // namespace gauss::gp
