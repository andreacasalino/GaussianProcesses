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
 * @brief https : // www.cs.toronto.edu/~duvenaud/cookbook/
 *
 */
class KernelFunction {
public:
  virtual ~KernelFunction() = default;

  KernelFunction(const KernelFunction &) = delete;
  KernelFunction &operator==(const KernelFunction &) = delete;

  virtual std::size_t numberOfParameters() const = 0;

  /**
   * @return the collection of tunable parameters, i.e. the ones that can be
   * tuned through training.
   */
  virtual std::vector<double> getParameters() const = 0;

  virtual void setParameters(const std::vector<double> &values) const = 0;

  /**
   * @brief evaluation should be reflexive: evaluate(a,b) = evaluate(b,a)
   *
   */
  virtual double evaluate(const Eigen::VectorXd &a,
                          const Eigen::VectorXd &b) const = 0;

  /**
   * @param a
   * @param b
   * @return The gradient of the kernel activation function
   */
  virtual std::vector<double> getGradient(const Eigen::VectorXd &a,
                                          const Eigen::VectorXd &b) const = 0;

protected:
  KernelFunction() = default;
};
using KernelFunctionPtr = std::unique_ptr<KernelFunction>;
} // namespace gauss::gp
