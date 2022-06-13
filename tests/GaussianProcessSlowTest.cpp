#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/Error.h>
#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "Grid.h"
#include "Utils.h"

#include <iostream>
#include <math.h>

TEST_CASE("Gaussian process predictions", "[gp_slow]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  const std::size_t input_size = GENERATE(2, 3);
  auto output_size = 2;

  Eigen::VectorXd min = Eigen::VectorXd::Ones(input_size);
  min *= -6.0;
  Eigen::VectorXd max = Eigen::VectorXd::Ones(input_size);
  max *= 6.0;
  EquispacedGrid grid(min, max, 10);

  GaussianProcess process(std::make_unique<SquaredExponential>(1.f, 0.5f),
                          input_size, output_size);
  process.setWhiteNoiseStandardDeviation(0.001);
  grid.gridFor([&process, &output_size](const Eigen::VectorXd &sample_in) {
    Eigen::VectorXd sample_out = Eigen::VectorXd::Ones(output_size);
    sample_out *= sin(sample_in.norm());
    process.getTrainSet().addSample(sample_in, sample_out);
  });

  for (std::size_t t = 0; t < 5; ++t) {
    const auto point = grid.at(grid.randomIndices());
    const auto point_prediction = process.predict2(point);
    const auto point_perturbed_prediction =
        process.predict2(point + 0.4 * grid.getDeltas());

    CHECK(point_prediction.mean.size() == output_size);
    Eigen::VectorXd expected_mean = Eigen::VectorXd::Ones(output_size);
    expected_mean *= sin(point.norm());
    CHECK(is_equal_vec(point_prediction.mean, expected_mean, 0.1));
    CHECK(point_prediction.covariance < point_perturbed_prediction.covariance);
  }
}

#include <TrainingTools/iterative/solvers/GradientDescend.h>

TEST_CASE("Check gradient direction", "[gp_slow]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  const std::size_t input_size = 3;
  const std::size_t output_size = 2; // GENERATE(1, 2);

  auto equispaced = GENERATE(true, false);
  std::vector<Eigen::VectorXd> samples;
  if (equispaced) {
    const Eigen::VectorXd max = 6.0 * Eigen::VectorXd::Ones(input_size);
    const Eigen::VectorXd min = (-6.0) * Eigen::VectorXd::Ones(input_size);
    EquispacedGrid grid(min, max, 5);
    grid.gridFor([&samples](const Eigen::VectorXd &sample_in) {
      const double value = sin(sample_in.norm());
      const Eigen::VectorXd sample_out =
          value * Eigen::VectorXd::Ones(static_cast<Eigen::Index>(output_size));
      auto &sample = samples.emplace_back(sample_in + sample_out);
      sample << sample_in, sample_out;
    });
  } else {
    auto samples_size = GENERATE(10, 50);
    for (std::size_t k = 0; k < samples_size; ++k) {
      Eigen::VectorXd sample_in(input_size);
      sample_in.setRandom();
      const double value = sin(sample_in.norm());
      const Eigen::VectorXd sample_out =
          value * Eigen::VectorXd::Ones(static_cast<Eigen::Index>(output_size));
      auto &sample = samples.emplace_back(sample_in + sample_out);
      sample << sample_in, sample_out;
    }
  }

  GaussianProcess process(std::make_unique<SquaredExponential>(2.0, 5.0),
                          input_size, output_size);
  for (const auto &sample : samples) {
    process.getTrainSet().addSample(sample);
  }

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

TEST_CASE("Train session the hyperparameters", "[gp_slow]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  const std::size_t input_size = 3;
  const std::size_t output_size = 2; // GENERATE(1, 2);

  const Eigen::VectorXd max = 6.0 * Eigen::VectorXd::Ones(input_size);
  const Eigen::VectorXd min = (-6.0) * Eigen::VectorXd::Ones(input_size);

  GaussianProcess process(std::make_unique<SquaredExponential>(2.0, 5.0),
                          input_size, output_size);

  EquispacedGrid grid(min, max, 5);
  grid.gridFor([&process](const Eigen::VectorXd &sample_in) {
    const double value = sin(sample_in.norm());
    const Eigen::VectorXd sample_out =
        value * Eigen::VectorXd::Ones(static_cast<Eigen::Index>(output_size));
    process.getTrainSet().addSample(sample_in, sample_out);
  });

  train::GradientDescendFixed trainer;
  trainer.setOptimizationStep(0.01);
  trainer.setMaxIterations(10);

  for (std::size_t k = 0; k < 3; ++k) {
    std::cout << process.getHyperParameters().transpose() << std::endl;
    std::cout << process.getHyperParametersGradient().transpose() << std::endl;
    const auto likelihood_prev = process.getLogLikelihood();
    process.train(trainer);
    const auto likelihood = process.getLogLikelihood();
    CHECK(likelihood_prev < likelihood);
  }
}
