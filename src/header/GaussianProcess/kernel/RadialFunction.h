/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>

namespace gauss::gp {
class RadialFunction;
using RadialFunctionPtr = std::unique_ptr<RadialFunction>;

class RadialFunction {
public:
  virtual ~RadialFunction() = default;

  const std::vector<double> &getParameters() const { return parameters; };
  void setParameters(const std::vector<double> &values);

  virtual RadialFunctionPtr copy() const = 0;

  virtual double evaluate(const double squared_distance) const = 0;

  virtual std::vector<double>
  getGradient(const double squared_distance) const = 0;

protected:
  RadialFunction(const std::vector<double> &params) : parameters(params){};

private:
  std::vector<double> parameters;
};

class RadialKernelFunction : public KernelFunction {
public:
  RadialKernelFunction(RadialFunctionPtr radial_function);

  KernelFunctionPtr copy() const final {
    return std::make_unique<RadialKernelFunction>(radial_function->copy());
  }

  std::size_t numberOfParameters() const final {
    return radial_function->getParameters().size();
  }

  std::vector<double> getParameters() const final {
    return radial_function->getParameters();
  }

  void setParameters(const std::vector<double> &values) final {
    radial_function->setParameters(values);
  }

  double evaluate(const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) const final;

  std::vector<double> getGradient(const Eigen::VectorXd &a,
                                  const Eigen::VectorXd &b) const final;

private:
  RadialFunctionPtr radial_function;
};
} // namespace gauss::gp
