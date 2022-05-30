/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/kernel/KernelFunction.h>

namespace gauss::gp {
KernelFunction::KernelFunction(const std::vector<ParameterPtr> &params)
    : parameters(params) {}

std::vector<ParameterCnstPtr> KernelFunction::getParameters() const {
  std::vector<ParameterCnstPtr> result;
  result.reserve(parameters.size());
  for (const auto &param : parameters) {
    result.push_back(param);
  }
  return result;
}
} // namespace gauss::gp
