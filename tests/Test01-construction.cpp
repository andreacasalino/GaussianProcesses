#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include <gtest/gtest.h>

template <typename ProcessT, typename... Args>
void expect_ctor_throw(Args... args) {
  try {
    ProcessT{std::forward<Args>(args)...};
  } catch (gauss::gp::Error) {
    return;
  } catch (const std::exception &e) {
    throw std::runtime_error("Invalid thrown exception type");
  }
}

gauss::gp::TrainSet make_train_set(const std::size_t out_size) {
  Eigen::VectorXd input(2);
  input << 1, 1;
  Eigen::VectorXd output(out_size);
  for (std::size_t o = 0; o < out_size; ++o) {
    output(o) = 1;
  }
  return gauss::gp::TrainSet{input, output};
}

TEST(Construction, GaussianProcess) {
  gauss::gp::GaussianProcess(
      std::make_unique<gauss::gp::SquaredExponential>(0, 1),
      static_cast<std::size_t>(1));

  gauss::gp::GaussianProcess(
      std::make_unique<gauss::gp::SquaredExponential>(0, 1),
      static_cast<std::size_t>(5));

  expect_ctor_throw<gauss::gp::GaussianProcess>(nullptr,
                                                static_cast<std::size_t>(1));

  expect_ctor_throw<gauss::gp::GaussianProcess>(
      std::make_unique<gauss::gp::SquaredExponential>(0, 1),
      static_cast<std::size_t>(0));

  gauss::gp::GaussianProcess(
      std::make_unique<gauss::gp::SquaredExponential>(0, 1), make_train_set(1));

  expect_ctor_throw<gauss::gp::GaussianProcess>(
      std::make_unique<gauss::gp::SquaredExponential>(0, 1), make_train_set(2));
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
