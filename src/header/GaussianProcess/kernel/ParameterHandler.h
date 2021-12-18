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
using Parameter = std::shared_ptr<double>;

class ParameterHandler {
public:
  virtual ~ParameterHandler() = default;

  virtual double evaluate_gradient(const Eigen::VectorXd &a,
                          const Eigen::VectorXd &b) const = 0;

  double getParameter() const { return *parameter; };
  void setParameter(const double value) { *parameter = value; };

protected:
  ParameterHandler(const Parameter& parameter) {
      this->parameter = parameter;
  };

private:
  Parameter parameter;
};
using ParameterHandlerPtr = std::unique_ptr<ParameterHandler>;
} // namespace gauss::gp
