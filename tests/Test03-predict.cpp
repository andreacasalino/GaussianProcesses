#include "Utils.h"
#include <GaussianProcess/kernel/SquaredExponential.h>
#include <gtest/gtest.h>

namespace gauss::gp::test {
template <std::size_t InputSize, std::size_t OutputSize>
class GaussianProcessPredictTest
    : public GaussianProcessTest<InputSize, OutputSize> {
public:
  GaussianProcessPredictTest()
      : GaussianProcessTest<InputSize, OutputSize>(
            std::make_unique<SquaredExponential>(1, 0.5)){};

  void SetUp() {
    auto samples = this->make_samples(5);

    this->pushSamples(samples.GetSamplesInput().GetSamples(),
                      samples.GetSamplesOutput().GetSamples());
  }

protected:
  void check_prediction() const {
    const auto &samples = this->getTrainSet()->GetSamplesInput().GetSamples();

    double prediction_covariance;
    for (const auto &sample : samples) {
      this->predict(sample, prediction_covariance);
      auto prediction_covariance_low = prediction_covariance;

      auto point = sample;
      {
        Eigen::VectorXd delta(sample.size());
        delta.setRandom();
        delta *= 0.05;
        point += delta;
      }
      this->predict(point, prediction_covariance);
      EXPECT_LE(prediction_covariance_low, prediction_covariance);
    }
  };
};
} // namespace gauss::gp::test

using Process3_2 = gauss::gp::test::GaussianProcessPredictTest<3, 2>;

TEST_F(Process3_2, prediction) { check_prediction(); }

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
