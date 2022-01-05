#include "Utils.h"
#include <gtest/gtest.h>

namespace gauss::gp::test {
template <std::size_t InputSize, std::size_t OutputSize>
class GaussianProcessUpdateTest
    : public GaussianProcessTest<InputSize, OutputSize> {
public:
  GaussianProcessUpdateTest() = default;

protected:
  void check_kernel_matrix() const {
    const auto &samples = this->getTrainSet()->GetSamplesInput().GetSamples();
    std::size_t expected_size = samples.size();
    Eigen::MatrixXd kernel = this->getCovariance();
    EXPECT_EQ(kernel.rows(), expected_size);
    EXPECT_EQ(kernel.cols(), expected_size);
    for (std::size_t r = 0; r < expected_size; ++r) {
      for (std::size_t c = 0; c < expected_size; ++c) {
        EXPECT_LE(abs(kernel(r, c) -
                      this->kernelFunction->evaluate(samples[r], samples[c])),
                  1e-3);
      }
    }
  };

  void check_output_samples_matrix() const {
    const auto &samples = this->getTrainSet()->GetSamplesOutput().GetSamples();
    std::size_t expected_size = samples.size();
    auto out_matrix = this->getSamplesOutputMatrix();
    EXPECT_EQ(out_matrix.rows(), expected_size);
    for (std::size_t r = 0; r < expected_size; ++r) {
      for (Eigen::Index i = 0; i < OutputSize; ++i) {
        EXPECT_EQ(out_matrix(r, i), samples[r](i));
      }
    }
  };

  void check_update_single_sample() {
    this->pushSample(this->make_sample_input(), this->make_sample_output());
    EXPECT_EQ(this->getCovariance().rows(), 1);
    EXPECT_EQ(this->getCovariance().cols(), 1);

    this->pushSample(this->make_sample_input(), this->make_sample_output());
    EXPECT_EQ(this->getCovariance().rows(), 2);
    EXPECT_EQ(this->getCovariance().cols(), 2);

    this->check_kernel_matrix();
    this->check_output_samples_matrix();

    this->clearSamples();
    EXPECT_THROW(this->getCovariance(), gauss::gp::Error);
    EXPECT_THROW(this->getCovarianceInv(), gauss::gp::Error);
    EXPECT_THROW(this->getCovarianceDeterminant(), gauss::gp::Error);
    EXPECT_THROW(this->getSamplesOutputMatrix(), gauss::gp::Error);

    this->pushSample(this->make_sample_input(), this->make_sample_output());
    EXPECT_EQ(this->getCovariance().rows(), 1);
    EXPECT_EQ(this->getCovariance().cols(), 1);

    this->check_kernel_matrix();
    this->check_output_samples_matrix();
  }

  void check_update_multiple_samples() {
    const std::size_t samples_numb = 5;

    auto samples1 = this->make_samples(samples_numb);

    this->pushSamples(samples1.GetSamplesInput().GetSamples(),
                      samples1.GetSamplesOutput().GetSamples());
    EXPECT_EQ(this->getCovariance().rows(), samples_numb);
    EXPECT_EQ(this->getCovariance().cols(), samples_numb);
    this->check_kernel_matrix();
    this->check_output_samples_matrix();

    auto samples2 = this->make_samples(samples_numb);

    this->pushSamples(samples1.GetSamplesInput().GetSamples(),
                      samples1.GetSamplesOutput().GetSamples());
    EXPECT_EQ(this->getCovariance().rows(), 2 * samples_numb);
    EXPECT_EQ(this->getCovariance().cols(), 2 * samples_numb);
    this->check_kernel_matrix();
    this->check_output_samples_matrix();
  }
};
} // namespace gauss::gp::test

using Process2_1 = gauss::gp::test::GaussianProcessUpdateTest<2, 1>;

TEST_F(Process2_1, noInit) {
  EXPECT_EQ(getTrainSet(), nullptr);
  EXPECT_THROW(getCovariance(), gauss::gp::Error);
  EXPECT_THROW(getCovarianceInv(), gauss::gp::Error);
  EXPECT_THROW(getCovarianceDeterminant(), gauss::gp::Error);
  EXPECT_THROW(getSamplesOutputMatrix(), gauss::gp::Error);
}

TEST_F(Process2_1, updateSingleSamples) { check_update_single_sample(); }

TEST_F(Process2_1, updateMultipleSamples) { check_update_multiple_samples(); }

using Process3_2 = gauss::gp::test::GaussianProcessUpdateTest<3, 2>;

TEST_F(Process3_2, updateSingleSamples) { check_update_single_sample(); }

TEST_F(Process3_2, updateMultipleSamples) { check_update_multiple_samples(); }

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
