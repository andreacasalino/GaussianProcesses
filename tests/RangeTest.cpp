#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "../samples/Ranges.h"
#include "Utils.h"

TEST_CASE("Linspace test", "[range]") {
  using namespace gauss::gp;
  using namespace gauss::gp::samples;
  using namespace gauss::gp::test;

  const std::size_t intervals = 20;

  Linspace range(-1.0, 1.0, intervals + 1);
  REQUIRE(std::abs(range.getDelta() - 2.0 / static_cast<double>(intervals)) <
          DEFAULT_TOLL);
  std::size_t counter = 0;
  for (; range(); ++range) {
    ++counter;
  }
  CHECK(counter == intervals + 1);
}

TEST_CASE("Grid 2d test", "[range]") {
  using namespace gauss::gp;
  using namespace gauss::gp::samples;
  using namespace gauss::gp::test;

  const std::size_t size = 3;

  Grid range(std::array<double, 2>{-1.0, 1.0}, std::array<double, 2>{-1.0, 1.0},
             size);
  std::vector<std::array<std::size_t, 2>> indices;
  std::size_t counter = 0;
  for (; range(); ++range) {
    indices.push_back(range.indices());
    ++counter;
  }
  CHECK(counter == size * size);
  CHECK(indices == std::vector<std::array<std::size_t, 2>>{{0, 0},
                                                           {0, 1},
                                                           {0, 2},
                                                           {1, 0},
                                                           {1, 1},
                                                           {1, 2},
                                                           {2, 0},
                                                           {2, 1},
                                                           {2, 2}});
}

TEST_CASE("Grid Rn test", "[range]") {
  using namespace gauss::gp;
  using namespace gauss::gp::samples;
  using namespace gauss::gp::test;

  auto size = 5;

  auto axis = GENERATE(2, 3, 4);
  std::vector<std::array<double, 2>> intervals;
  for (std::size_t k = 0; k < axis; ++k) {
    intervals.push_back({-1.0, 1.0});
  }

  GridMultiDimensional range(size, intervals);

  std::size_t counter = 0;
  for (; range(); ++range) {
    ++counter;
  }
  CHECK(counter == static_cast<std::size_t>(pow(size, axis)));
}
