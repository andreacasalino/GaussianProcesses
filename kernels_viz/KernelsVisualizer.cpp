#include <GaussianProcess/kernel/Linear.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "KernelVisualizer.h"

#include <fstream>
#include <iostream>

int main() {
  const std::size_t size = 100;

  nlohmann::json kernels_log = nlohmann::json::array();

  ///////////// squared exponential function /////////////
  kernels_log.push_back(gauss::gp::make_kernel_viz_log(
      size, gauss::gp::SquaredExponential(1.0, 0.1),
      "squared exponential d = 0.1", "squared_exp"));
  std::cout << "squared exponential  d = 0.1" << std::endl;

  kernels_log.push_back(gauss::gp::make_kernel_viz_log(
      size, gauss::gp::SquaredExponential(1.0, 0.5),
      "squared exponential d = 0.5", "squared_exp"));
  std::cout << "squared exponential  d = 0.5" << std::endl;

  kernels_log.push_back(gauss::gp::make_kernel_viz_log(
      size, gauss::gp::SquaredExponential(1.0, 1.0),
      "squared exponential d = 1.0", "squared_exp"));
  std::cout << "squared exponential  d = 1.0" << std::endl;

  ///////////// linear function /////////////
  kernels_log.push_back(gauss::gp::make_kernel_viz_log(
      size, gauss::gp::LinearFunction(1.0, 1.0, Eigen::VectorXd::Zero(1)),
      "linear function mean = 0", "linear"));
  std::cout << "linear function mean = 0" << std::endl;

  kernels_log.push_back(gauss::gp::make_kernel_viz_log(
      size,
      gauss::gp::LinearFunction(1.0, 1.0, -Eigen::VectorXd::Ones(1) * 0.7),
      "linear function mean = -0.7", "linear"));
  std::cout << "linear function mean = -0.7" << std::endl;

  kernels_log.push_back(gauss::gp::make_kernel_viz_log(
      size, gauss::gp::LinearFunction(1.0, 1.0, Eigen::VectorXd::Ones(1) * 0.7),
      "linear function mean = 0.7", "linear"));
  std::cout << "linear function mean = 0.7" << std::endl;

  std::ofstream stream("kernels_log.json");
  stream << kernels_log.dump();

  return EXIT_SUCCESS;
}