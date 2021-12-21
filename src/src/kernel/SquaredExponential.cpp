/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/kernel/SquaredExponential.h>

namespace gauss::gp {
namespace {
double evaluate_exp(const Eigen::VectorXd &a, const Eigen::VectorXd &b,
                    const double teta1) {
  auto delta = a;
  delta -= b;
  double result = exp(-teta1 * delta.squaredNorm());
  return result;
};

class Teta0Handler : public ParameterHandler {
public:
  Teta0Handler(const Parameter &teta0, const Parameter &teta1)
      : ParameterHandler(teta0) {
    this->teta1 = teta1;
  };

  double evaluate_gradient(const Eigen::VectorXd &a,
                           const Eigen::VectorXd &b) const override {
    return evaluate_exp(a, b, *teta1);
  };

private:
  Parameter teta1;
};

class Teta1Handler : public ParameterHandler {
public:
  Teta1Handler(const Parameter &teta0, const Parameter &teta1)
      : ParameterHandler(teta1) {
    this->teta0 = teta0;
  };

  double evaluate_gradient(const Eigen::VectorXd &a,
                           const Eigen::VectorXd &b) const override {
    return -*teta0 * evaluate_exp(a, b, getParameter()) * (a - b).squaredNorm();
  };

private:
  Parameter teta0;
};
} // namespace

SquaredExponential::SquaredExponential(const double teta0, const double teta1) {
  this->teta0 = std::make_shared<double>(teta0);
  this->teta1 = std::make_shared<double>(teta1);
}

double SquaredExponential::evaluate(const Eigen::VectorXd &a,
                                    const Eigen::VectorXd &b) const {
  return *teta0 * evaluate_exp(a, b, *teta1);
};

std::unique_ptr<KernelFunction> SquaredExponential::copy() const {
  return std::make_unique<SquaredExponential>(*teta0, *teta1);
};

std::vector<ParameterHandlerPtr> SquaredExponential::getParameters() const {
  std::vector<ParameterHandlerPtr> result;
  result.reserve(2);
  result.emplace_back(std::make_unique<Teta0Handler>(teta0, teta1));
  result.emplace_back(std::make_unique<Teta1Handler>(teta0, teta1));
  return result;
};
} // namespace gauss::gp
