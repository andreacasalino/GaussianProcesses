#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "Utils.h"

#include <math.h>

namespace {
Eigen::VectorXd make_vec(const double val) {
  Eigen::VectorXd result(1);
  result << val;
  return result;
}

class EquispacedGrid {
public:
  EquispacedGrid(const Eigen::VectorXd &min_corner,
                 const Eigen::VectorXd &max_corner, const std::size_t size)
      : size(size) {
    if (0 == size) {
      throw std::runtime_error{"Invalid grid size"};
    }
    deltas = max_corner - min_corner;
    deltas /= static_cast<double>(size - 1);
  }

  void
  gridFor(const std::function<void(const Eigen::VectorXd &)> &predicate) const {
    gridFor_(predicate, {});
  }

  const Eigen::VectorXd &getDeltas() const { return deltas; }

  Eigen::VectorXd at(const std::vector<std::size_t> &indices) const {
    Eigen::VectorXd result(deltas.size());
    for (std::size_t k = 0; k < indices.size(); ++k) {
      result(k) = indices[k] * deltas(k);
    }
    return result;
  }

  std::vector<std::size_t> randomIndices() const {
    std::vector<std::size_t> result;
    for (std::size_t k = 0; k < deltas.size(); ++k) {
      result.push_back(rand() % size);
    }
    return result;
  }

private:
  void gridFor_(const std::function<void(const Eigen::VectorXd &)> &predicate,
                const std::vector<std::size_t> &cumulated_indices) const {
    if (cumulated_indices.size() == deltas.size()) {
      predicate(at(cumulated_indices));
      return;
    }
    for (std::size_t k = 0; k < size; ++k) {
      auto indices = cumulated_indices;
      indices.push_back(k);
      gridFor_(predicate, indices);
    }
  }

  const std::size_t size;
  Eigen::VectorXd deltas;
};
} // namespace

TEST_CASE("Gaussian process predictions 1d", "[predict]") {
  using namespace gauss::gp;

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
