#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include <TrainingTools/iterative/solvers/GradientDescend.h>

#include "Grid.h"

TEST_CASE("Train the hyperparameters", "[train]") {
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