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
/**
 * @brief Handler of a single kernel function parameter.
 *
 */
class Parameter {
public:
  virtual ~Parameter() = default;

  Parameter(const Parameter &) = delete;
  Parameter &operator==(const Parameter &) = delete;

  /**
   * @param a
   * @param b
   * @return The gradient of the kernel activation function
   */
  virtual double evaluate_gradient(const Eigen::VectorXd &a,
                                   const Eigen::VectorXd &b) const = 0;

  double getParameter() const { return parameter_value; };
  void setParameter(const double value) { parameter_value = value; };

protected:
  Parameter(const double parameter) : parameter_value(parameter){};

private:
  double parameter_value;
};
using ParameterPtr = std::shared_ptr<Parameter>;
using ParameterCnstPtr = std::shared_ptr<const Parameter>;

/**
 * @brief https : // www.cs.toronto.edu/~duvenaud/cookbook/
 *
 */
class KernelFunction {
public:
  virtual ~KernelFunction() = default;

  KernelFunction(const KernelFunction &) = delete;
  KernelFunction &operator==(const KernelFunction &) = delete;

  /**
   * @brief evaluation should be reflexive: evaluate(a,b) = evaluate(b,a)
   *
   */
  virtual double evaluate(const Eigen::VectorXd &a,
                          const Eigen::VectorXd &b) const = 0;

  /**
   * @return the collection of tunable parameters, i.e. the ones that can be
   * tuned through training.
   */
  std::vector<ParameterCnstPtr> getParameters() const;

protected:
  KernelFunction(const std::vector<ParameterPtr> &params);

private:
  std::vector<ParameterPtr> parameters;
};
using KernelFunctionPtr = std::unique_ptr<KernelFunction>;
} // namespace gauss::gp
