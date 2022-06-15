#include <GaussianProcess/GaussianProcess.h>

// #include <GaussianProcess/kernel/Linear.h>
// #include <GaussianProcess/kernel/PeriodicFunction.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

void make_gradient_viz_log(nlohmann::json &recipient,
                           gauss::gp::KernelFunctionPtr kernel,
                           const Eigen::VectorXd &min_corner,
                           const Eigen::VectorXd &max_corner,
                           const std::size_t size);

Eigen::VectorXd make_vector(const std::vector<double> &vals);

int main() {
  const std::size_t size = 50;

  struct KernelAndTitle {
    gauss::gp::KernelFunctionPtr kernel;
    Eigen::VectorXd min_corner;
    Eigen::VectorXd max_corner;
  };

  std::unordered_map<std::string, KernelAndTitle> cases;

  cases.emplace(
      "exponential",
      KernelAndTitle{std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0),
                     make_vector({-2, -2}), make_vector({2, 2})});

  nlohmann::json gradients_log;
  for (auto &[tag, data] : cases) {
    std::cout << tag;
    make_gradient_viz_log(gradients_log[tag], std::move(data.kernel),
                          data.min_corner, data.max_corner, size);
    std::cout << " done" << std::endl;
  }
  std::ofstream stream("gradients_log.json");
  stream << gradients_log.dump();

  std::cout << "python3 GradientVisualizer.py" << std::endl;

  return EXIT_SUCCESS;
}

Eigen::VectorXd make_vector(const std::vector<double> &vals) {
  Eigen::VectorXd result(vals.size());
  result.setZero();
  for (std::size_t k = 0; k < vals.size(); ++k) {
    result(k) = vals[k];
  }
  return result;
}

namespace {
class Range {
public:
  Range(const double min, const double max, const std::size_t size)
      : val(min), delta((max - min) / static_cast<double>(size - 1)),
        size(size), counter(0) {}

  std::size_t getCounter() const { return counter; }
  Range &operator++() {
    ++counter;
    val += delta;
    return *this;
  }
  double operator()() const { return val; };
  bool ongoing() const { return counter < size; };

private:
  double val;
  const double delta;
  const std::size_t size;
  std::size_t counter;
};

using Matrix = std::vector<std::vector<double>>;
Matrix make_matrix(const std::size_t size) {
  Matrix result;
  for (std::size_t r = 0; r < size; ++r) {
    auto &row = result.emplace_back();
    for (std::size_t c = 0; c < size; ++c) {
      row.push_back(0);
    }
  }
  return result;
}
} // namespace

void make_gradient_viz_log(nlohmann::json &recipient,
                           gauss::gp::KernelFunctionPtr kernel,
                           const Eigen::VectorXd &min_corner,
                           const Eigen::VectorXd &max_corner,
                           const std::size_t size) {
  gauss::gp::GaussianProcess process(std::move(kernel), 3, 1);
  for (std::size_t k = 0; k < 30; ++k) {
    Eigen::VectorXd input_sample(3);
    input_sample.setRandom();
    input_sample *= 6.0;
    process.getTrainSet().addSample(input_sample,
                                    make_vector({sin(input_sample.norm())}));
  }

  Matrix log_lkl = make_matrix(size);
  std::vector<Matrix> grad_lkl;
  for (std::size_t k = 0; k < process.getKernelFunction().numberOfParameters();
       ++k) {
    grad_lkl.emplace_back(make_matrix(size));
  }

  for (Range range_theta_0 = Range{min_corner(0), max_corner(0), size};
       range_theta_0.ongoing(); ++range_theta_0) {
    for (Range range_theta_1 = Range{min_corner(0), max_corner(0), size};
         range_theta_1.ongoing(); ++range_theta_1) {
      Eigen::VectorXd par(2);
      par << range_theta_0(), range_theta_1();
      process.setHyperParameters(par);

      log_lkl[range_theta_0.getCounter()][range_theta_1.getCounter()] =
          process.getLogLikelihood();

      auto grad = process.getHyperParametersGradient();
      for (std::size_t g = 0; g < grad.size(); ++g) {
        grad_lkl[g][range_theta_0.getCounter()][range_theta_1.getCounter()] =
            grad(g);
      }
    }
  }

  recipient["likelihood"] = log_lkl;
  recipient["likelihood_gradient"] = grad_lkl;
}
