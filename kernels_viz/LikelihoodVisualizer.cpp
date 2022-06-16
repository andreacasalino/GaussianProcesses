#include <GaussianProcess/GaussianProcess.h>

// #include <GaussianProcess/kernel/Linear.h>
#include <GaussianProcess/kernel/PeriodicFunction.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "../samples/LogUtils.h"
#include "../samples/Ranges.h"

#include <algorithm>
#include <iostream>
#include <unordered_map>
#include <variant>

using ValueOrRange = std::variant<double, std::array<double, 2>>;

class ParametersRange {
public:
  ParametersRange(const std::vector<ValueOrRange> &values,
                  const std::size_t size);

  Eigen::VectorXd eval() const;

private:
  const std::array<Eigen::Index, 2> mutable_indices;
  Eigen::VectorXd values_;

public:
  std::unique_ptr<gauss::gp::samples::Grid> mutable_parameters;
};

void make_gradient_viz_log(nlohmann::json &recipient,
                           gauss::gp::KernelFunctionPtr kernel,
                           ParametersRange &range);

int main() {
  const std::size_t size = 100;

  struct KernelAndTitle {
    gauss::gp::KernelFunctionPtr kernel;
    std::vector<ValueOrRange> parameters_range;
  };

  std::unordered_map<std::string, KernelAndTitle> cases;

  cases.emplace(
      "exponential",
      KernelAndTitle{
          std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0),
          {std::array<double, 2>{0.5, 2.0}, std::array<double, 2>{0.5, 10.0}}});

  // cases.emplace("periodic",
  //               KernelAndTitle{std::make_unique<gauss::gp::PeriodicFunction>(
  //                                  1.0, 1.0, 0.1),
  //                              {std::array<double, 2>{0.5, 2.0},
  //                               std::array<double, 2>{0.5, 10.0}, 0.1}});

  nlohmann::json gradients_log;
  for (auto &[tag, data] : cases) {
    std::cout << tag;
    ParametersRange params(data.parameters_range, size);
    make_gradient_viz_log(gradients_log[tag], std::move(data.kernel), params);
    std::cout << " done" << std::endl;
  }
  gauss::gp::samples::print(gradients_log, "likelihhod_log.json");

  std::cout << "python3 LikelihoodVisualizer.py" << std::endl;

  return EXIT_SUCCESS;
}

using Matrix = std::vector<std::vector<double>>;

Matrix make_square(const std::size_t size) {
  Matrix result;
  for (std::size_t k = 0; k < size; ++k) {
    result.emplace_back().resize(size);
  }
  return result;
}

std::array<Eigen::Index, 2>
find_mutable(const std::vector<ValueOrRange> &values) {
  struct Visitor {
    mutable Eigen::Index counter = 0;
    mutable std::vector<Eigen::Index> result;

    void operator()(const double val) const { ++counter; }
    void operator()(const std::array<double, 2> val) const {
      result.push_back(counter);
      ++counter;
    }

  } visitor;

  for (const auto &value : values) {
    std::visit(visitor, value);
  }

  if (2 != visitor.result.size()) {
    throw std::runtime_error{"Invalid parameters range"};
  }
  return {visitor.result.front(), visitor.result.back()};
}

ParametersRange::ParametersRange(const std::vector<ValueOrRange> &values,
                                 const std::size_t size)
    : mutable_indices(find_mutable(values)), values_(values.size()) {
  for (Eigen::Index k = 0; k < values.size(); ++k) {
    if (std::find(mutable_indices.begin(), mutable_indices.end(), k) ==
        mutable_indices.end()) {
      values_(k) = std::get<double>(values[k]);
    }
  }
  const auto &par_0 =
      std::get<std::array<double, 2>>(values[mutable_indices.front()]);
  const auto &par_1 =
      std::get<std::array<double, 2>>(values[mutable_indices.back()]);
  mutable_parameters = std::make_unique<gauss::gp::samples::Grid>(
      std::array<double, 2>{par_0.front(), par_1.front()},
      std::array<double, 2>{par_0.back(), par_1.back()}, size);
}

Eigen::VectorXd ParametersRange::eval() const {
  auto result = values_;
  auto mut_values = mutable_parameters->eval();
  result(mutable_indices[0]) = mut_values(0);
  result(mutable_indices[1]) = mut_values(1);
  return result;
}

void make_gradient_viz_log(nlohmann::json &recipient,
                           gauss::gp::KernelFunctionPtr kernel,
                           ParametersRange &range) {
  gauss::gp::GaussianProcess process(std::move(kernel), 3, 1);
  for (std::size_t k = 0; k < 30; ++k) {
    Eigen::VectorXd input_sample(3);
    input_sample.setRandom();
    input_sample *= 6.0;
    Eigen::VectorXd output_sample(1);
    output_sample << sin(input_sample.norm());
    process.getTrainSet().addSample(input_sample, output_sample);
  }

  Matrix likelihood = make_square(range.mutable_parameters->getSize());
  std::vector<Matrix> likelihood_gradient;
  for (std::size_t k = 0; k < process.getKernelFunction().numberOfParameters();
       ++k) {
    likelihood_gradient.push_back(
        make_square(range.mutable_parameters->getSize()));
  }

  auto &hyper_paramters_grid = *range.mutable_parameters;
  for (; hyper_paramters_grid(); ++hyper_paramters_grid) {
    const auto param = range.eval();
    process.setHyperParameters(param);
    const auto indices = hyper_paramters_grid.indices();
    likelihood[indices[0]][indices[1]] = process.getLogLikelihood();
    const auto grad = process.getHyperParametersGradient();
    for (std::size_t k = 0; k < grad.size(); ++k) {
      likelihood_gradient[k][indices[0]][indices[1]] = grad(k);
    }
  }

  recipient["likelihood"] = likelihood;
  recipient["likelihood_gradient"] = likelihood_gradient;
}
