#include "Utils.h"
#include <gtest/gtest.h>

namespace gauss::gp::test {
template <std::size_t InputSize, std::size_t OutputSize>
class GaussianProcessPredictTest
    : public GaussianProcessTest<InputSize, OutputSize> {
public:
  GaussianProcessPredictTest() = default;

  void SetUp() {
    auto samples = this->make_samples(5);

    this->pushSamples(samples.GetSamplesInput().GetSamples(),
                      samples.GetSamplesOutput().GetSamples());
  }

protected:
  void check_prediction() const {
    const auto &samples = this->getTrainSet()->GetSamplesInput().GetSamples();

    Eigen::VectorXd prediction_mean;
    double prediction_covariance;

    this->predict(samples.front(), prediction_mean, prediction_covariance);
    auto prediction_covariance_low = prediction_covariance;

    for (std::size_t k = 0; k < 5; ++k) {
      this->predict(this->make_sample_input(), prediction_mean,
                    prediction_covariance);
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
