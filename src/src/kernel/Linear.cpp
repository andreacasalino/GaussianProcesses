/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/kernel/Linear.h>

namespace gauss::gp {
namespace {
Eigen::VectorXd delta(const std::vector<Parameter> &mean,
                      const Eigen::VectorXd &x) {
  Eigen::VectorXd result(x.size());
  for (std::size_t k = 0; k < mean.size(); ++k) {
    result(k) = *mean[k] - x(static_cast<Eigen::Index>(k));
  }
  return result;
};

class Teta0Handler : public ParameterHandler {
public:
  Teta0Handler(const Parameter &teta0) : ParameterHandler(teta0){};

  double evaluate_gradient(const Eigen::VectorXd &a,
                           const Eigen::VectorXd &b) const override {
    return 2.0 * getParameter();
  };
};

class Teta1Handler : public ParameterHandler {
public:
  Teta1Handler(const Parameter &teta1) : ParameterHandler(teta1){};

  double evaluate_gradient(const Eigen::VectorXd &a,
                           const Eigen::VectorXd &b) const override {
    Eigen::VectorXd delta_a = delta(mean, a);
    Eigen::VectorXd delta_b = delta(mean, b);
    return 2.0 * getParameter() * delta_a.dot(delta_b);
  };

private:
  std::vector<Parameter> mean;
};

class MeanHandler : public ParameterHandler {
public:
  MeanHandler(const Parameter &teta1, const Parameter &mean_i,
              const Eigen::Index i_pos)
      : ParameterHandler(mean_i), i_pos(i_pos) {
    this->teta1 = teta1;
  };

  double evaluate_gradient(const Eigen::VectorXd &a,
                           const Eigen::VectorXd &b) const override {
    return *teta1 * *teta1 * (getParameter() - a(i_pos) - b(i_pos));
  };

private:
  Parameter teta1;
  const Eigen::Index i_pos;
};
} // namespace

namespace {
Eigen::VectorXd null_mean(const Eigen::Index &size) {
  Eigen::VectorXd result(size);
  result.setZero();
  return result;
}
} // namespace

LinearFunction::LinearFunction(const double teta0, const double teta1,
                               const std::size_t space_size)
    : LinearFunction(teta0, teta1,
                     null_mean(static_cast<Eigen::Index>(space_size))) {}

LinearFunction::LinearFunction(const double teta0, const double teta1,
                               const Eigen::VectorXd &mean) {
  this->teta0 = std::make_shared<double>(teta0);
  this->teta1 = std::make_shared<double>(teta1);
  this->mean.reserve(mean.size());
  for (Eigen::Index i = 0; i < mean.size(); ++i) {
    this->mean.push_back(std::make_shared<double>(mean(i)));
  }
}

LinearFunction::LinearFunction(const LinearFunction &o) {
  this->teta0 = std::make_shared<double>(*o.teta0);
  this->teta1 = std::make_shared<double>(*o.teta1);
  this->mean.reserve(mean.size());
  for (const auto &mean_val : mean) {
    this->mean.push_back(std::make_shared<double>(*mean_val));
  }
}

double LinearFunction::evaluate(const Eigen::VectorXd &a,
                                const Eigen::VectorXd &b) const {
  Eigen::VectorXd delta_a = delta(mean, a);
  Eigen::VectorXd delta_b = delta(mean, b);
  return *teta0 * *teta0 + *teta1 * *teta1 * delta_a.dot(delta_b);
};

std::unique_ptr<KernelFunction> LinearFunction::copy() const {
  std::unique_ptr<LinearFunction> result;
  result.reset(new LinearFunction(*this));
  return result;
};

std::vector<ParameterHandlerPtr> LinearFunction::getParameters() const {
  std::vector<ParameterHandlerPtr> result;
  result.reserve(2 + mean.size());
  result.emplace_back(std::make_unique<Teta0Handler>(teta0));
  result.emplace_back(std::make_unique<Teta1Handler>(teta1));
  Eigen::Index i_pos = 0;
  for (std::size_t k = 0; k < mean.size(); ++k, ++i_pos) {
    result.emplace_back(std::make_unique<MeanHandler>(teta1, mean[k], i_pos));
  }
  return result;
};
} // namespace gauss::gp
