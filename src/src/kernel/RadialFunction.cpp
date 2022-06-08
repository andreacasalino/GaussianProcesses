/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/Error.h>
#include <GaussianProcess/kernel/RadialFunction.h>

namespace gauss::gp {
void RadialFunction::setParameters(const std::vector<double> &values) {
  if (values.size() != parameters.size()) {
    throw Error{"Invalid parameters for RadialFunction"};
  }
  parameters = values;
};

RadialKernelFunction::RadialKernelFunction(RadialFunctionPtr radial_function) {
  if (nullptr == radial_function) {
    throw Error{"Invalid null radial function"};
  }
  this->radial_function = std::move(radial_function);
}

namespace {
double squared_distance(const Eigen::VectorXd &a, const Eigen::VectorXd &b) {
  auto distance = a;
  distance -= b;
  return distance.squaredNorm();
}
} // namespace

double RadialKernelFunction::evaluate(const Eigen::VectorXd &a,
                                      const Eigen::VectorXd &b) const {
  return radial_function->evaluate(squared_distance(a, b));
}

std::vector<double>
RadialKernelFunction::getGradient(const Eigen::VectorXd &a,
                                  const Eigen::VectorXd &b) const {
  return radial_function->getGradient(squared_distance(a, b));
}
} // namespace gauss::gp
