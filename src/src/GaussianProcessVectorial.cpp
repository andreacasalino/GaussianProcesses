/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/GaussianProcessVectorial.h>

namespace gauss::gp {
GaussianProcessVectorial::GaussianProcessVectorial(
    KernelFunctionPtr kernel, const std::size_t input_space_size,
    const std::size_t output_space_size)
    : GaussianProcessBase(std::move(kernel), input_space_size,
                          output_space_size) {}

GaussianProcessVectorial::GaussianProcessVectorial(
    KernelFunctionPtr kernel, gauss::gp::TrainSet train_set)
    : GaussianProcessBase(std::move(kernel), std::move(train_set)) {}

std::vector<gauss::GaussianDistribution>
GaussianProcessVectorial::predict(const Eigen::VectorXd &point) const {
  double prediction_covariance;
  Eigen::VectorXd prediction_mean =
      GaussianProcessBase::predict(point, prediction_covariance);
  Eigen::MatrixXd prediction_covariance_mat(1, 1);
  prediction_covariance_mat << prediction_covariance;
  std::vector<gauss::GaussianDistribution> result;
  result.reserve(prediction_mean.size());
  for (Eigen::Index k = 0; k < prediction_mean.size(); ++k) {
    Eigen::VectorXd temp(1);
    temp << prediction_mean(k);
    result.emplace_back(temp, prediction_covariance_mat);
  }
  return result;
}

GaussianProcessVectorial::Prediction
GaussianProcessVectorial::predict2(const Eigen::VectorXd &point) const {
  double covariance;
  Eigen::VectorXd mean = GaussianProcessBase::predict(point, covariance);
  return Prediction{mean, covariance};
}

GaussianDistribution
GaussianProcessVectorial::predict3(const Eigen::VectorXd &point) const {
  double covariance;
  Eigen::VectorXd mean = GaussianProcessBase::predict(point, covariance);
  Eigen::MatrixXd big_cov(mean.size(), mean.size());
  big_cov.setZero();
  for (Eigen::Index i = 0; i < big_cov.size(); ++i) {
    big_cov(i, i) = covariance;
  }
  return GaussianDistribution{mean, big_cov};
}
} // namespace gauss::gp
