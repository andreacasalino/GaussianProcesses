#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/kernel/CompositeFunction.h>
#include <GaussianProcess/kernel/Linear.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "Utils.h"
#include <Eigen/Dense>

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

TEST_CASE("Check kernel functions", "[kernel-functions]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  auto function = GENERATE(
      make_kernel_function<SquaredExponential>(1, 0.2),
      make_kernel_function<LinearFunction>(0.5, 1.0, 4),
      make_kernel_function<LinearFunction>(0.5, 1.0, Eigen::VectorXd::Ones(4)),
      make_kernel_function<Summation>(
          std::make_unique<SquaredExponential>(1, 0.2),
          std::make_unique<LinearFunction>(0.5, 1.0, 4)),
      std::make_shared<Product>(std::make_unique<SquaredExponential>(1, 0.2),
                                std::make_unique<LinearFunction>(0.5, 1.0, 4)));

  const std::size_t size = 30;
  const auto samples = gauss::gp::test::make_samples(size, 4);
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
