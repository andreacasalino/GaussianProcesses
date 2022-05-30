/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/Error.h>
#include <GaussianProcess/kernel/CombinedFunction.h>

namespace gauss::gp {
void CombinedFunction::addElement(KernelFunctionPtr element) {
  if (nullptr == element) {
    throw Error{"can't add null element to CombinedFunction kernel"};
  }
  elements.emplace_back(std::move(element));
  params_numb += element->numberOfParameters();
}

CombinedFunction::CombinedFunction(KernelFunctionPtr first,
                                   KernelFunctionPtr second) {
  addElement(std::move(first));
  addElement(std::move(second));
}

std::vector<double> CombinedFunction::getParameters() const {
  std::vector<double> result;
  for (const auto &element : elements) {
    for (const auto &val : element->getParameters()) {
      result.push_back(val);
    }
  }
  return result;
}

void CombinedFunction::setParameters(const std::vector<double> &values) const {
  if (values.size() != params_numb) {
    throw Error{"Invalid parameters"};
  }
  using Iter = std::vector<double>::const_iterator;
  Iter cursor = values.begin();
  for (const auto &element : elements) {
    auto cursor_end = cursor;
    std::advance(cursor_end, element->numberOfParameters());
    element->setParameters(std::vector<double>{cursor, cursor_end});
    cursor = cursor_end;
  }
}

double CombinedFunction::evaluate(const Eigen::VectorXd &a,
                                  const Eigen::VectorXd &b) const {
  double result = 0;
  for (const auto &element : elements) {
    result += element->evaluate(a, b);
  }
  return result;
}

std::vector<double>
CombinedFunction::get_gradient(const Eigen::VectorXd &a,
                               const Eigen::VectorXd &b) const {
  std::vector<double> result;
  for (const auto &element : elements) {
    for (const auto &val : element->get_gradient(a, b)) {
      result.push_back(val);
    }
  }
  return result;
}
} // namespace gauss::gp
