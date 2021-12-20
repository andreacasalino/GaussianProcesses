#include "Utils.h"
#include <gtest/gtest.h>

using Process2_1 = gauss::gp::test::GaussianProcessTest<2, 1>;

TEST_F(Process2_1, noInit) {
  EXPECT_EQ(getTrainSet(), nullptr);
  EXPECT_THROW(getKernel(), gauss::gp::Error);
  EXPECT_THROW(getKernelInverse(), gauss::gp::Error);
  EXPECT_THROW(getCovarianceDeterminant(), gauss::gp::Error);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
