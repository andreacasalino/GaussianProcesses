#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/kernel/CompositeFunction.h>
#include <GaussianProcess/kernel/Linear.h>
#include <GaussianProcess/kernel/PeriodicFunction.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "../samples/Ranges.h"

#include <Eigen/Dense>

#include <iostream>
#include <unordered_map>

namespace {
Eigen::VectorXcd compute_eigvalues(const Eigen::MatrixXd &subject) {
  Eigen::EigenSolver<Eigen::MatrixXd> solver(subject);
  return solver.eigenvalues();
}

bool are_real(const Eigen::VectorXcd &eigen_values) {
  for (Eigen::Index k = 0; k < eigen_values.size(); ++k) {
    if (abs(eigen_values(k).imag()) > 1e-4) {
      return false;
    }
  }
  return true;
}

bool are_positive(const Eigen::VectorXd &eigen_values) {
  for (Eigen::Index k = 0; k < eigen_values.size(); ++k) {
    if (eigen_values(k) < -1e-4) {
      return false;
    }
  }
  return true;
}

template <typename KernelT, typename... Args>
std::shared_ptr<gauss::gp::KernelFunction> make_kernel_function(Args... args) {
  return std::make_shared<KernelT>(std::forward<Args>(args)...);
}
} // namespace

#include "Utils.h"

TEST_CASE("Check kernel functions", "[kernel-functions]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  std::unordered_map<std::string, KernelFunctionPtr> labels_and_functions;
  labels_and_functions.emplace("exponential",
                               std::make_unique<SquaredExponential>(1, 0.2));
  labels_and_functions.emplace("linear 0 mean",
                               std::make_unique<LinearFunction>(0.5, 1.0, 4));
  labels_and_functions.emplace(
      "linear",
      std::make_unique<LinearFunction>(0.5, 1.0, Eigen::VectorXd::Ones(4)));
  // labels_and_functions.emplace("periodic",
  //                              std::make_unique<PeriodicFunction>(1, 0.2,
  //                              0.1)); // periodic stand alone should never be
  //                              used
  labels_and_functions.emplace(
      "summation of linear and exponential",
      std::make_unique<Summation>(
          std::make_unique<SquaredExponential>(1, 0.2),
          std::make_unique<LinearFunction>(0.5, 1.0, 4)));
  labels_and_functions.emplace(
      "product of linear and exponential",
      std::make_unique<Product>(std::make_unique<SquaredExponential>(1, 0.2),
                                std::make_unique<LinearFunction>(0.5, 1.0, 4)));
  labels_and_functions.emplace(
      "product of periodic and exponential",
      std::make_unique<Product>(
          std::make_unique<SquaredExponential>(1, 0.2),
          std::make_unique<PeriodicFunction>(1, 0.2, 0.1)));

  const std::size_t size = 30;
  const auto samples = gauss::gp::test::make_samples(size, 1.0, 4);

  for (const auto &[label, function] : labels_and_functions) {
    std::cout << "checking " << label << std::endl;
    Eigen::MatrixXd kernel_matrix(size, size);
    for (Eigen::Index r = 0; r < size; ++r) {
      for (Eigen::Index c = 0; c < size; ++c) {
        kernel_matrix(r, c) =
            function->evaluate(samples[static_cast<std::size_t>(r)],
                               samples[static_cast<std::size_t>(c)]);
      }
    }
    CHECK(is_symmetric(kernel_matrix));
    const auto eig_values = compute_eigvalues(kernel_matrix);
    CHECK(are_real(eig_values));
    CHECK(are_positive(eig_values.real()));
  }
}
