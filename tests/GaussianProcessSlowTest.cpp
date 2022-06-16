#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/Error.h>
#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "../samples/Ranges.h"
#include "Utils.h"

#include <iostream>
#include <math.h>

TEST_CASE("Gaussian process predictions", "[gp_slow]") {
  using namespace gauss::gp;
  using namespace gauss::gp::samples;

  const std::size_t input_size = GENERATE(2, 3);
  auto output_size = 2;

  test::GridMultiDimensional grid(
      10, test::make_intervals(-6.0 * Eigen::VectorXd::Ones(input_size),
                               6.0 * Eigen::VectorXd::Ones(input_size)));

  GaussianProcess process(std::make_unique<SquaredExponential>(1.f, 0.5f),
                          input_size, output_size);
  process.setWhiteNoiseStandardDeviation(0.001);
  for (; grid(); ++grid) {
    const auto sample_in = grid.eval();
    Eigen::VectorXd sample_out = Eigen::VectorXd::Ones(output_size);
    sample_out *= sin(sample_in.norm());
    process.getTrainSet().addSample(sample_in, sample_out);
  }

  const auto &samples_in = process.getTrainSet().GetSamplesInput();
  for (std::size_t t = 0; t < 10; ++t) {
    Eigen::VectorXd point;
    {
      auto it = samples_in.begin();
      std::advance(it, rand() % samples_in.size());
      point = *it;
    }
    const auto point_prediction = process.predict2(point);
    const auto point_perturbed_prediction =
        process.predict2(point + 0.4 * grid.getDeltas());

    CHECK(point_prediction.mean.size() == output_size);
    Eigen::VectorXd expected_mean = Eigen::VectorXd::Ones(output_size);
    expected_mean *= sin(point.norm());
    CHECK(test::is_equal_vec(point_prediction.mean, expected_mean, 0.1));
    CHECK(point_prediction.covariance < point_perturbed_prediction.covariance);
  }
}

TEST_CASE("Check gradient and training", "[gp_slow]") {
  using namespace gauss::gp;

  const std::size_t input_size = 3;
  const std::size_t output_size = GENERATE(1, 2);

  auto equispaced = GENERATE(true, false);
  std::vector<Eigen::VectorXd> samples;
  if (equispaced) {
    test::GridMultiDimensional grid(
        10, test::make_intervals(-6.0 * Eigen::VectorXd::Ones(input_size),
                                 6.0 * Eigen::VectorXd::Ones(input_size)));
    for (; grid(); ++grid) {
      const auto sample_in = grid.eval();
      const double value = sin(sample_in.norm());
      const Eigen::VectorXd sample_out =
          value * Eigen::VectorXd::Ones(static_cast<Eigen::Index>(output_size));
      samples.emplace_back(sample_in + sample_out) << sample_in, sample_out;
    }
  } else {
    auto samples_size = GENERATE(10, 50);
    for (std::size_t k = 0; k < samples_size; ++k) {
      Eigen::VectorXd sample_in(input_size);
      sample_in.setRandom();
      sample_in *= 6.0;
      const double value = sin(sample_in.norm());
      const Eigen::VectorXd sample_out =
          value * Eigen::VectorXd::Ones(static_cast<Eigen::Index>(output_size));
      auto &sample = samples.emplace_back(sample_in.size() + sample_out.size());
      sample << sample_in, sample_out;
    }
  }

  GaussianProcess process(std::make_unique<SquaredExponential>(2.0, 5.0),
                          input_size, output_size);
  for (const auto &sample : samples) {
    process.getTrainSet().addSample(sample);
  }

  SECTION("Gradient direction") {
    auto invert = GENERATE(true, false);

    const auto initial_L = process.getLogLikelihood();
    auto grad = process.getHyperParametersGradient();

    if (invert) {
      process.setHyperParameters(process.getHyperParameters() - grad * 0.01);
      const auto new_L = process.getLogLikelihood();
      CHECK(initial_L > new_L);
    } else {
      process.setHyperParameters(process.getHyperParameters() + grad * 0.01);
      const auto new_L = process.getLogLikelihood();
      CHECK(initial_L < new_L);
    }
  }

  SECTION("Simple train") {
    for (std::size_t k = 0; k < 3; ++k) {
      const auto likelihood_prev = process.getLogLikelihood();
      for (std::size_t j = 0; j < 10; ++j) {
        const auto param = process.getHyperParameters();
        const auto grad = process.getHyperParametersGradient();
        process.setHyperParameters(param + grad * 0.1);
      }
      const auto likelihood = process.getLogLikelihood();
      CHECK(likelihood_prev < likelihood);
    }
  }
}
