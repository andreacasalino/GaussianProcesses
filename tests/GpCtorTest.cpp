#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/Error.h>
#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include "Utils.h"

TEST_CASE("construction of Gaussian processes", "[ctor]") {
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
