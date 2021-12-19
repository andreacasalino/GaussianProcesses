#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/GaussianProcessVectorial.h>
#include <GaussianProcess/kernel/LinearFunction.h>

#include <gtest/gtest.h>

#define EXPECT_CTOR_THROW(ProcessT, __VA_ARGS__)                               \
  try {                                                                        \
    ProcessT{__VA_ARGS__};                                                     \
  } catch (gauss::gp::Error) {                                                 \
    return;                                                                    \
  } catch (const std::exception &e) {                                          \
    throw std::runtime_error("Invalid thrown exception type");                 \
  }

TEST(Construction, GaussianProcess) {
  gauss::gp::GaussianProcess(std::make_unique<gauss::gp::LinearFunction>(0, 1),
                             static_cast<std::size_t>(1));

  gauss::gp::GaussianProcess(std::make_unique<gauss::gp::LinearFunction>(0, 1),
                             static_cast<std::size_t>(5));

  EXPECT_CTOR_THROW(gauss::gp::GaussianProcess, nullptr,
                    static_cast<std::size_t>(1));

  //   {
  //     std::unique_ptr<gauss::gp::KernelFunction> function =
  //         std::make_unique<gauss::gp::LinearFunction>(0, 1);
  //     expect_ctor_throw(std::move(function), static_cast<std::size_t>(0));
  //   }
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
