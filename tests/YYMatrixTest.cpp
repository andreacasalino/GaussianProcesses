#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "Utils.h"

TEST_CASE("Check YY matrix computation", "[YY_matrix]") {
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
