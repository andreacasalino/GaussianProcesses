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

  GaussianProcessTest(KernelFunctionPtr kernel_function)
      : GaussianProcessBase(std::move(kernel_function), InputSize,
                            OutputSize){};

protected:
  Eigen::VectorXd make_sample_input() const {
    Eigen::VectorXd result(InputSize);
    result.setRandom();
    return result;
  };
  Eigen::VectorXd make_sample_output() const {
    Eigen::VectorXd result(OutputSize);
    result.setRandom();
    return result;
  };

  TrainSet make_samples(const std::size_t samples_numb) const {
    std::vector<Eigen::VectorXd> input_samples;
    input_samples.reserve(samples_numb);
    std::vector<Eigen::VectorXd> output_samples;
    output_samples.reserve(samples_numb);
    for (std::size_t s = 0; s < samples_numb; ++s) {
      input_samples.emplace_back(make_sample_input());
      output_samples.emplace_back(make_sample_output());
    }
    return gauss::gp::TrainSet{input_samples, output_samples};
  };
};
} // namespace gauss::gp::test
