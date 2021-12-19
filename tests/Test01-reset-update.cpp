#include <gtest/gtest.h>

TEST(Range, binaryGroupSmall) { 
    int var = 1;
    EXPECT_EQ(var, 1);
}


int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
