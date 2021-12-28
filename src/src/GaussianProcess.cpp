/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/GaussianProcess.h>

namespace gauss::gp {
GaussianProcess::GaussianProcess(KernelFunctionPtr kernel,
                                 const std::size_t input_space_size)
    : GaussianProcessBase(std::move(kernel), input_space_size, 1) {}

GaussianProcess::GaussianProcess(KernelFunctionPtr kernel,
                                 gauss::gp::TrainSet train_set)
    : GaussianProcessBase(std::move(kernel), std::move(train_set)) {
  if (getOutputStateSpaceSize() != 1) {
    throw gauss::gp::Error("Invalid output samples");
  }
}

gauss::GaussianDistribution
GaussianProcess::predict(const Eigen::VectorXd &point) const {
  double prediction_covariance;
  Eigen::VectorXd prediction_mean =
      GaussianProcessBase::predict(point, prediction_covariance).transpose();
  Eigen::MatrixXd prediction_covariance_mat(1, 1);
  prediction_covariance_mat << prediction_covariance;
  return gauss::GaussianDistribution(prediction_mean,
                                     prediction_covariance_mat);
}

GaussianProcess::Prediction
GaussianProcess::predict2(const Eigen::VectorXd &point) const {
  double prediction_covariance;
  Eigen::VectorXd prediction_mean =
      GaussianProcessBase::predict(point, prediction_covariance);
  return Prediction{prediction_mean(0), prediction_covariance};
}
} // namespace gauss::gp
