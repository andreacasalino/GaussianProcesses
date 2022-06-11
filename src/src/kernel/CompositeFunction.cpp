/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/Error.h>
#include <GaussianProcess/kernel/CompositeFunction.h>

namespace gauss::gp {
CompositeFunction::CompositeFunction(KernelFunctionPtr first,
                                     KernelFunctionPtr second) {
  addElement(std::move(first));
  addElement(std::move(second));
}

CompositeFunction::CompositeFunction(const CompositeFunction &o) {
  for (auto &element : o.elements) {
    addElement(element->copy());
  }
}

void CompositeFunction::addElement(KernelFunctionPtr element) {
  if (nullptr == element) {
    throw Error{"can't add null element to CombinedFunction kernel"};
  }
  elements.emplace_back(std::move(element));
  params_numb += element->numberOfParameters();
}

std::vector<double> CompositeFunction::getParameters() const {
  std::vector<double> result;
  for (const auto &element : elements) {
    for (const auto &val : element->getParameters()) {
      result.push_back(val);
    }
  }
  return result;
}

void CompositeFunction::setParameters(const std::vector<double> &values) {
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

Summation::Summation(KernelFunctionPtr first, KernelFunctionPtr second)
    : CompositeFunction(std::move(first), std::move(second)) {}

Summation::Summation(const Summation &o) : CompositeFunction(o) {}

KernelFunctionPtr Summation::copy() const {
  std::unique_ptr<Summation> result;
  result.reset(new Summation{*this});
  return result;
}

double Summation::evaluate(const Eigen::VectorXd &a,
                           const Eigen::VectorXd &b) const {
  double result = 0;
  for (const auto &element : getElements()) {
    result += element->evaluate(a, b);
  }
  return result;
}

std::vector<double> Summation::getGradient(const Eigen::VectorXd &a,
                                           const Eigen::VectorXd &b) const {
  std::vector<double> result;
  for (const auto &element : getElements()) {
    for (const auto &val : element->getGradient(a, b)) {
      result.push_back(val);
    }
  }
  return result;
}

Product::Product(KernelFunctionPtr first, KernelFunctionPtr second)
    : CompositeFunction(std::move(first), std::move(second)) {}

Product::Product(const Product &o) : CompositeFunction(o) {}

KernelFunctionPtr Product::copy() const {
  std::unique_ptr<Product> result;
  result.reset(new Product{*this});
  return result;
}

double Product::evaluate(const Eigen::VectorXd &a,
                         const Eigen::VectorXd &b) const {
  double result = 1.0;
  for (const auto &element : getElements()) {
    result *= element->evaluate(a, b);
  }
  return result;
}

std::vector<double> Product::getGradient(const Eigen::VectorXd &a,
                                         const Eigen::VectorXd &b) const {
  std::vector<double> result;
  double product = 1.0;
  for (const auto &element : getElements()) {
    const auto element_eval = element->evaluate(a, b);
    product *= element_eval;
    for (const auto &val : element->getGradient(a, b)) {
      result.push_back(val / element_eval);
    }
  }
  for (auto &val : result) {
    val *= product;
  }
  return result;
}
} // namespace gauss::gp
