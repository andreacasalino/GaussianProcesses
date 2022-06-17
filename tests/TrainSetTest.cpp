#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/Error.h>
#include <GaussianProcess/TrainSet.h>

#include <memory>

namespace {
gauss::TrainSet make_samples(const std::size_t samples_numb,
                             const Eigen::Index sample_size) {
  std::vector<Eigen::VectorXd> result;
  for (std::size_t k = 0; k < samples_numb; ++k) {
    result.emplace_back(sample_size).setZero();
  }
  return result;
}
} // namespace

TEST_CASE("Train set ctor", "[train_set]") {
  using namespace gauss::gp;

  TrainSet{3, 1};
  TrainSet{1, 3};

  CHECK_THROWS_AS(std::make_unique<TrainSet>(3, 0), Error);
  CHECK_THROWS_AS(std::make_unique<TrainSet>(0, 3), Error);

  TrainSet{make_samples(10, 5), make_samples(10, 3)};
}

TEST_CASE("Samples addition", "[train_set]") {
  using namespace gauss::gp;

  TrainSet train_set{5, 3};
  const auto sample_in = Eigen::VectorXd::Zero(5);
  const auto sample_out = Eigen::VectorXd::Ones(3);

  SECTION("split vectors") {
    train_set.addSample(sample_in, sample_out);

    CHECK_THROWS_AS(train_set.addSample(Eigen::VectorXd::Zero(4), sample_out),
                    Error);
    CHECK_THROWS_AS(train_set.addSample(Eigen::VectorXd::Zero(6), sample_out),
                    Error);
    CHECK_THROWS_AS(train_set.addSample(sample_in, Eigen::VectorXd::Zero(2)),
                    Error);
    CHECK_THROWS_AS(train_set.addSample(sample_in, Eigen::VectorXd::Zero(4)),
                    Error);
  }

  SECTION("merged vector") {
    Eigen::VectorXd sample(8);
    sample << sample_in, sample_out;
    train_set.addSample(sample);

    const auto &samples_in = train_set.GetSamplesInput();
    REQUIRE(samples_in.size() == 1);
    CHECK(samples_in.front() == sample_in);

    const auto &samples_out = train_set.GetSamplesOutput();
    REQUIRE(samples_out.size() == 1);
    CHECK(samples_out.front() == sample_out);

    CHECK_THROWS_AS(train_set.addSample(Eigen::VectorXd::Zero(7)), Error);
    CHECK_THROWS_AS(train_set.addSample(Eigen::VectorXd::Zero(9)), Error);
  }
}
