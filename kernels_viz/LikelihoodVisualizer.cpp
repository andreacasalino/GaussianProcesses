#include <GaussianProcess/GaussianProcess.h>

// #include <GaussianProcess/kernel/Linear.h>
// #include <GaussianProcess/kernel/PeriodicFunction.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "../samples/LogUtils.h"
#include "../samples/Ranges.h"

#include <iostream>
#include <unordered_map>

void make_gradient_viz_log(nlohmann::json &recipient,
                           gauss::gp::KernelFunctionPtr kernel,
                           const Eigen::VectorXd &min_corner,
                           const Eigen::VectorXd &max_corner,
                           const std::size_t size);

int main() {
  const std::size_t size = 100;

  struct KernelAndTitle {
    gauss::gp::KernelFunctionPtr kernel;
    Eigen::VectorXd min_corner;
    Eigen::VectorXd max_corner;
  };

  std::unordered_map<std::string, KernelAndTitle> cases;

  cases.emplace(
      "exponential",
      KernelAndTitle{std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0),
                     make_vector({0.2, 10.0}), make_vector({10.0, 10.0})});

  nlohmann::json gradients_log;
  for (auto &[tag, data] : cases) {
    std::cout << tag;
    make_gradient_viz_log(gradients_log[tag], std::move(data.kernel),
                          data.min_corner, data.max_corner, size);
    std::cout << " done" << std::endl;
  }
  gauss::gp::samples::print(gradients_log, "gradients_log.json");

  std::cout << "python3 LikelihoodVisualizer.py" << std::endl;

  return EXIT_SUCCESS;
}

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
