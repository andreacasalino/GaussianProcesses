#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "Grid.h"
#include "Utils.h"

#include <math.h>

namespace {
Eigen::VectorXd make_vec(const double val) {
  Eigen::VectorXd result(1);
  result << val;
  return result;
}

} // namespace

TEST_CASE("Gaussian process predictions 1d", "[predict]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  EquispacedGrid grid(make_vec(-6.0), make_vec(6.0), 20);

  GaussianProcessScalar<1> process(
      std::make_unique<SquaredExponential>(1.f, 0.5f));
  grid.gridFor([&process](const Eigen::VectorXd &sample_in) {
    const auto sample_out = make_vec(sin(sample_in(0)));
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
    CHECK(std::abs(point_prediction.mean(0) - sin(point(0))) < test::TOLL);
    CHECK(point_prediction.covariance < point_perturbed_prediction.covariance);
  }
}

TEST_CASE("Gaussian process predictions 3d", "[predict]") {
  using namespace gauss::gp;
  using namespace gauss::gp::test;

  Eigen::VectorXd min(3);
  min << -6.0, -6.0, -6.0;
  Eigen::VectorXd max(3);
  max << 6.0, 6.0, 6.0;
  EquispacedGrid grid(min, max, 10);

  SECTION("scalar output") {
    GaussianProcessScalar<3> process(
        std::make_unique<SquaredExponential>(1.f, 0.5f));
    grid.gridFor([&process](const Eigen::VectorXd &sample_in) {
      const auto sample_out = make_vec(sin(sample_in.norm()));
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
      CHECK(std::abs(point_prediction.mean(0) - sin(point.norm())) <
            test::TOLL);
      CHECK(point_prediction.covariance <
            point_perturbed_prediction.covariance);
    }
  }

  SECTION("vectorial output") {
    auto make_out = [](const Eigen::VectorXd &sample_in) {
      const auto val = sin(sample_in.norm());
      Eigen::VectorXd result(3);
      result << val, val, val;
      return result;
    };

    GaussianProcessVectorial<3, 3> process(
        std::make_unique<SquaredExponential>(1.f, 0.5f));
    grid.gridFor([&process, &make_out](const Eigen::VectorXd &sample_in) {
      process.getTrainSet().addSample(sample_in, make_out(sample_in));
    });

    for (std::size_t t = 0; t < 10; ++t) {
      const auto point = grid.at(grid.randomIndices());
      const auto point_prediction = process.predict2(point);
      const auto point_perturbed_prediction =
          process.predict2(point + 0.5 * grid.getDeltas());

      CHECK(0 < point_prediction.covariance);
      CHECK(0 < point_perturbed_prediction.covariance);
      CHECK(point_prediction.mean.size() == 3);
      CHECK(test::is_equal_vec(point_prediction.mean, make_out(point)));
      CHECK(point_prediction.covariance <
            point_perturbed_prediction.covariance);
    }
  }
}
