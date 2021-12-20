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

protected:
  TrainSet make_samples(const std::size_t samples_numb) {
    std::vector<Eigen::VectorXd> input_samples;
    input_samples.reserve(samples_numb);
    std::vector<Eigen::VectorXd> output_samples;
    output_samples.reserve(samples_numb);
    for (std::size_t s = 0; s < samples_numb; ++s) {
      input_samples.emplace_back(InputSize);
      input_samples.back().setRandom();
      output_samples.emplace_back(OutputSize);
      output_samples.back().setRandom();
    }
    return gauss::gp::TrainSet{input_samples, output_samples};
  };
};
} // namespace gauss::gp::test
