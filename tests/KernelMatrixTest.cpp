// test kernel correctness
// test kernel covariance decomposition
// test kernel covariance is spd
// test kernel isr ecomputed when changing kernel function

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "Utils.h"

TEST_CASE("Check covariance computation", "[kernel_matrix]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  const std::size_t samples_numb = 10;

  auto kernel_function = std::make_unique<SquaredExponential>(1.f, 1.f);

  GaussianProcess process(kernel_function->copy(), 3, 1);
  const auto samples = test::make_samples(samples_numb, 4);
  for (const auto &sample : samples) {
    process.getTrainSet().addSample(sample);
  }

  const auto kernel_cov = process.getCovariance();
  REQUIRE(kernel_cov.rows() == samples_numb);
  REQUIRE(kernel_cov.cols() == samples_numb);
  CHECK(is_symmetric(kernel_cov));
  // check kernel_cov was correctly computed
  for (Eigen::Index r = 0; r < kernel_cov.rows(); ++r) {
    for (Eigen::Index c = r; c < kernel_cov.cols(); ++c) {
      auto sample_r = samples[static_cast<std::size_t>(r)].block(0, 0, 3, 1);
      auto sample_c = samples[static_cast<std::size_t>(c)].block(0, 0, 3, 1);
      CHECK(abs(kernel_cov(r, c) -
                kernel_function->evaluate(sample_r, sample_c)) < TOLL);
    }
  }

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
