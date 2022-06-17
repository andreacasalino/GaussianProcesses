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
class KernelFunction;
using KernelFunctionPtr = std::unique_ptr<KernelFunction>;

/**
 * @brief https : // www.cs.toronto.edu/~duvenaud/cookbook/
 *
 */
class KernelFunction {
public:
  virtual ~KernelFunction() = default;

  virtual KernelFunctionPtr copy() const = 0;

  KernelFunction(const KernelFunction &) = delete;
  KernelFunction &operator==(const KernelFunction &) = delete;

  /**
   * @return the number of hyperparameters pertaining to this kernel function
   */
  virtual std::size_t numberOfParameters() const = 0;

  /**
   * @return the current values of the hyperparameters
   */
  virtual std::vector<double> getParameters() const = 0;

  /**
   * @brief sets the hyperparameters values.
   */
  virtual void setParameters(const std::vector<double> &values) = 0;

  /**
   * @brief evaluation is expected be reflexive: evaluate(a,b) = evaluate(b,a)
   *
   */
  virtual double evaluate(const Eigen::VectorXd &a,
                          const Eigen::VectorXd &b) const = 0;

  /**
   * @return The gradient of the kernel activation function w.r.t. the
   * hyperparameters
   */
  virtual std::vector<double> getGradient(const Eigen::VectorXd &a,
                                          const Eigen::VectorXd &b) const = 0;

protected:
  KernelFunction() = default;
};
} // namespace gauss::gp
