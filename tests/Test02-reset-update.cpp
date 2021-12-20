#include "Utils.h"
#include <gtest/gtest.h>

using Process2_1 = gauss::gp::test::GaussianProcessTest<2, 1>;

TEST_F(Process2_1, noInit) {
  EXPECT_EQ(getTrainSet(), nullptr);
  EXPECT_THROW(getCovariance(), gauss::gp::Error);
  EXPECT_THROW(getCovarianceInv(), gauss::gp::Error);
  EXPECT_THROW(getCovarianceDeterminant(), gauss::gp::Error);
  EXPECT_THROW(getSamplesOutputMatrix(), gauss::gp::Error);
}

Eigen::VectorXd make_sample(const Eigen::Index size) {
  Eigen::VectorXd result(size);
  result.setRandom();
  return result;
}

TEST_F(Process2_1, updateSingleSample) {
  pushSample(make_sample(2), make_sample(1));
  EXPECT_EQ(getCovariance().rows(), 1);
  EXPECT_EQ(getCovariance().cols(), 1);

  pushSample(make_sample(2), make_sample(1));
  EXPECT_EQ(getCovariance().rows(), 2);
  EXPECT_EQ(getCovariance().cols(), 2);

  clearSamples();
  EXPECT_THROW(getCovariance(), gauss::gp::Error);
  EXPECT_THROW(getCovarianceInv(), gauss::gp::Error);
  EXPECT_THROW(getCovarianceDeterminant(), gauss::gp::Error);
  EXPECT_THROW(getSamplesOutputMatrix(), gauss::gp::Error);

  pushSample(make_sample(2), make_sample(1));
  EXPECT_EQ(getCovariance().rows(), 1);
  EXPECT_EQ(getCovariance().cols(), 1);
}

std::vector<Eigen::VectorXd> make_samples(const Eigen::Index size,
                                          const std::size_t samples) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples);
  for (std::size_t s = 0; s < samples; ++s) {
    result.emplace_back(make_sample(size));
  }
  return result;
}

TEST_F(Process2_1, updateMultipleSamples) {
  const std::size_t samples_numb = 5;

  pushSamples(make_samples(2, samples_numb), make_samples(1, samples_numb));
  EXPECT_EQ(getCovariance().rows(), samples_numb);
  EXPECT_EQ(getCovariance().cols(), samples_numb);

  pushSamples(make_samples(2, samples_numb), make_samples(1, samples_numb));
  EXPECT_EQ(getCovariance().rows(), 2 * samples_numb);
  EXPECT_EQ(getCovariance().cols(), 2 * samples_numb);

  clearSamples();
  EXPECT_THROW(getCovariance(), gauss::gp::Error);
  EXPECT_THROW(getCovarianceInv(), gauss::gp::Error);
  EXPECT_THROW(getCovarianceDeterminant(), gauss::gp::Error);
  EXPECT_THROW(getSamplesOutputMatrix(), gauss::gp::Error);

  pushSample(make_sample(2), make_sample(1));
  pushSamples(make_samples(2, samples_numb), make_samples(1, samples_numb));
  EXPECT_EQ(getCovariance().rows(), 1 + samples_numb);
  EXPECT_EQ(getCovariance().cols(), 1 + samples_numb);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
