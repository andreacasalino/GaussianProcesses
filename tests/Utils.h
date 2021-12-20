#include <GaussianProcess/components/GaussianProcessBase.h>
#include <GaussianProcess/kernel/LinearFunction.h>
#include <gtest/gtest.h>

namespace gauss::gp::test {
template <std::size_t InputSize, std::size_t OutputSize>
class GaussianProcessTest : public GaussianProcessBase, public ::testing::Test {
public:
  GaussianProcessTest()
      : GaussianProcessBase(std::make_unique<LinearFunction>(1, 1), InputSize,
                            OutputSize){};
};
} // namespace gauss::gp::test
