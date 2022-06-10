#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/Error.h>
#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

TEST_CASE("Construction of Gaussian processes", "[gp]") {
  using namespace gauss::gp;

  GaussianProcess{std::make_unique<SquaredExponential>(1.0, 0.5), 5, 2};

  CHECK_THROWS_AS(std::make_unique<GaussianProcess>(nullptr, 5, 2), Error);

  CHECK_THROWS_AS(std::make_unique<GaussianProcess>(
                      std::make_unique<SquaredExponential>(1.0, 0.5), 0, 2),
                  Error);

  CHECK_THROWS_AS(std::make_unique<GaussianProcess>(
                      std::make_unique<SquaredExponential>(1.0, 0.5), 5, 0),
                  Error);
}

#include "Utils.h"

TEST_CASE("Check covariance computation", "[gp]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  const std::size_t samples_numb = 10;

  auto kernel_function = std::make_unique<SquaredExponential>(1.f, 1.f);

  const std::size_t input_size = 3;

  GaussianProcess process(kernel_function->copy(), input_size, 1);
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
      auto sample_r =
          samples[static_cast<std::size_t>(r)].block(0, 0, input_size, 1);
      auto sample_c =
          samples[static_cast<std::size_t>(c)].block(0, 0, input_size, 1);
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

TEST_CASE("Check YY matrices computation", "[gp]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  const std::size_t samples_numb = 10;

  const std::size_t input_size = 3;
  auto output_size = GENERATE(2, 3, 4);

  GaussianProcess process(std::make_unique<SquaredExponential>(1.f, 1.f),
                          input_size, output_size);

  const auto samples_in = test::make_samples(samples_numb, input_size);
  const auto samples_out = test::make_samples(samples_numb, output_size);
  for (std::size_t k = 0; k < samples_numb; ++k) {
    process.getTrainSet().addSample(samples_in[k], samples_out[k]);
  }

  SECTION("YY train matrix") {
    const auto YY_train_matrix = process.getYYtrain();
    REQUIRE(YY_train_matrix.rows() == samples_numb);
    REQUIRE(YY_train_matrix.cols() == samples_numb);
    CHECK(is_symmetric(YY_train_matrix));
    for (Eigen::Index r = 0; r < YY_train_matrix.rows(); ++r) {
      for (Eigen::Index c = r; c < YY_train_matrix.cols(); ++c) {
        auto sample_r = samples_out[static_cast<std::size_t>(r)];
        auto sample_c = samples_out[static_cast<std::size_t>(c)];
        CHECK(abs(sample_r.dot(sample_c) - YY_train_matrix(r, c)) < TOLL);
      }
    }
  }

  SECTION("YY predict matrix") {
    const auto YY_predict_matrix = process.getYYpredict();
    REQUIRE(YY_predict_matrix.rows() == output_size);
    REQUIRE(YY_predict_matrix.cols() == samples_numb);
    for (Eigen::Index k = 0; k < samples_numb; ++k) {
      CHECK(is_equal_vec(YY_predict_matrix.col(k), samples_out[k]));
    }
  }
}
