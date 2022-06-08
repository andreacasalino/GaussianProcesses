#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <GaussianProcess/../../src/Common.h>
#include <GaussianProcess/Error.h>

namespace {
double eval(const Eigen::Index row, const Eigen::Index col) {
  return static_cast<double>(row) + static_cast<double>(col);
}

bool is_well_formed(const Eigen::MatrixXd &mat,
                    const Eigen::Index expected_size) {
  if ((expected_size != mat.rows()) || (expected_size != mat.cols())) {
    return false;
  }
  for (Eigen::Index r = 0; r < mat.rows(); ++r) {
    for (Eigen::Index c = 0; c < mat.cols(); ++c) {
      if (eval(r, c) != mat(r, c)) {
        return false;
      }
    }
  }
  return true;
}
} // namespace

TEST_CASE("Extendable matrix resize", "[resize]") {
  using namespace gauss::gp;

  SymmetricResizableMatrix matrix(eval);

  CHECK(matrix.getSize() == 0);
  CHECK(matrix.getComputedSize() == 0);

  std::size_t size = 0;
  for (std::size_t k = 0; k < 3; ++k) {
    const auto old_size = size;
    size += 5;
    matrix.resize(size);
    CHECK(matrix.getSize() == size);
    CHECK(matrix.getComputedSize() == old_size);

    CHECK(is_well_formed(matrix.access(), size));
    CHECK(matrix.getSize() == size);
    CHECK(matrix.getComputedSize() == size);
  }

  matrix.resize(0);
  CHECK(matrix.getSize() == size);
  CHECK(matrix.getComputedSize() == 0);
  CHECK_THROWS_AS(matrix.resize(10), Error);
  CHECK(is_well_formed(matrix.access(), size));
}
