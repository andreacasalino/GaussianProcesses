#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/Error.h>
#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "Grid.h"
#include "Utils.h"

#include <math.h>

TEST_CASE("Gaussian process predictions", "[gp_slow]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  const std::size_t input_size = GENERATE(1, 3);
  auto output_size = GENERATE(1, 2);

  Eigen::VectorXd min = Eigen::VectorXd::Ones(input_size);
  min *= -6.0;
  Eigen::VectorXd max = Eigen::VectorXd::Ones(input_size);
  max *= 6.0;
  EquispacedGrid grid(min, max, 10);

  GaussianProcess process(std::make_unique<SquaredExponential>(1.f, 0.5f),
                          input_size, output_size);
  grid.gridFor([&process, &output_size](const Eigen::VectorXd &sample_in) {
    Eigen::VectorXd sample_out = Eigen::VectorXd::Ones(output_size);
    sample_out *= sin(sample_in.norm());
    process.getTrainSet().addSample(sample_in, sample_out);
  });

  for (std::size_t t = 0; t < 10; ++t) {
    const auto point = grid.at(grid.randomIndices());
    const auto point_prediction = process.predict2(point);
    const auto point_perturbed_prediction =
        process.predict2(point + 0.5 * grid.getDeltas());

    CHECK(0 < point_prediction.covariance);
    CHECK(0 < point_perturbed_prediction.covariance);
    CHECK(point_prediction.mean.size() == 1);
    Eigen::VectorXd expected_mean = Eigen::VectorXd::Ones(output_size);
    expected_mean *= sin(point.norm());
    CHECK(is_equal_vec(point_prediction.mean, expected_mean));
    CHECK(point_prediction.covariance < point_perturbed_prediction.covariance);
  }
}

#include <TrainingTools/iterative/solvers/GradientDescend.h>

TEST_CASE("Train the hyperparameters", "[gp_slow]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  Eigen::VectorXd min(3);
  min << -6.0, -6.0, -6.0;
  Eigen::VectorXd max(3);
  max << 6.0, 6.0, 6.0;
  EquispacedGrid grid(min, max, 10);

  GaussianProcessVectorial<3, 2> process(
      std::make_unique<SquaredExponential>(1.f, 0.5f));
  grid.gridFor([&process](const Eigen::VectorXd &sample_in) {
    const double value = sin(sample_in.norm());
    Eigen::VectorXd sample_out(2);
    sample_out << value, value;
    process.getTrainSet().addSample(sample_in, sample_out);
  });

  train::GradientDescendFixed trainer;
  trainer.setOptimizationStep(0.01f);
  trainer.setMaxIterations(10);

  for (std::size_t k = 0; k < 3; ++k) {
    const auto likelihood_prev = process.getLogLikelihood();
    process.train(trainer);
    const auto likelihood = process.getLogLikelihood();
    CHECK(likelihood_prev < likelihood);
  }
}
