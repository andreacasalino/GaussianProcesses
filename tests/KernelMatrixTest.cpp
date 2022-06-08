// test kernel correctness
// test kernel covariance decomposition
// test kernel covariance is spd
// test kernel isr ecomputed when changing kernel function

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "Utils.h"

namespace {
static constexpr double TOLL = 1e-4;

bool is_zeros(const Eigen::MatrixXd &subject) {
  for (Eigen::Index r = 0; r < subject.rows(); ++r) {
    for (Eigen::Index c = 0; c < subject.cols(); ++c) {
      if (std::abs(subject(r, c)) > TOLL) {
        return false;
      }
    }
  }
  return true;
}

bool is_equal(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
  return is_zeros(a - b);
}

bool is_symmetric(const Eigen::MatrixXd &subject) {
  return is_equal(subject, subject.transpose());
}

bool is_inverse(const Eigen::MatrixXd &subject,
                const Eigen::MatrixXd &candidate) {
  return is_equal(subject * candidate,
                  Eigen::MatrixXd::Identity(subject.rows(), subject.cols()));
}
} // namespace

TEST_CASE("Check covariance computation", "[kernel_matrix]") {
  using namespace gauss::gp;

  const std::size_t samples_numb = 10;

  GaussianProcess process(std::make_unique<SquaredExponential>(1.f, 1.f), 3, 1);
  for (const auto &sample : test::make_samples(samples_numb, 4)) {
    process.getTrainSet().addSample(sample);
  }

  const auto kernel_cov = process.getCovariance();

  // check sizes
  REQUIRE(kernel_cov.rows() == samples_numb);
  REQUIRE(kernel_cov.cols() == samples_numb);
  CHECK(is_symmetric(kernel_cov));

  const auto decomposition = process.getCovarianceDecomposition();
  // check eigvals are positive
  for (const auto &eig_val : decomposition.eigenValues) {
    CHECK(TOLL < eig_val);
  }
  CHECK(decomposition.eigenVectors.rows() == samples_numb);
  CHECK(decomposition.eigenVectors.cols() == samples_numb);
  // check eigvectors are rotation matrix
  CHECK(is_inverse(decomposition.eigenVectors,
                   decomposition.eigenVectors.transpose()));
  // check decomposition
  CHECK(is_equal(kernel_cov,
                 Eigen::MatrixXd(decomposition.eigenVectors *
                                 decomposition.eigenValues.asDiagonal() *
                                 decomposition.eigenVectors.transpose())));

  CHECK(is_inverse(kernel_cov, process.getCovarianceInv()));
}
