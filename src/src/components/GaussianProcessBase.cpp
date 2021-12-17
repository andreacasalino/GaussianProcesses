/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/components/GaussianProcessBase.h>

namespace gauss::gp {
GaussianProcessBase::GaussianProcessBase(KernelFunctionPtr kernel,
                                         const std::size_t input_space_size,
                                         const std::size_t output_space_size)
    : InputOutputSizeAwareBase(input_space_size, output_space_size)
    , KernelAware(std::move(kernel)) {
}

GaussianProcessBase::GaussianProcessBase(KernelFunctionPtr kernel, gauss::gp::TrainSet train_set)
    : GaussianProcessBase(std::move(kernel), train_set.getInputStateSpaceSize(), train_set.getOutputStateSpaceSize()) {
    this->samples = std::make_unique<gauss::gp::TrainSet>(std::move(train_set));
    updateKernel();
    updateSamplesOutputMatrix();
}

void GaussianProcessBase::pushSample_(const Eigen::VectorXd &input_sample,
                                      const Eigen::VectorXd &output_sample) {
  if (input_sample.size() != getInputStateSpaceSize()) {
    throw Error("Invalid size for input sample");
  }
  if (output_sample.size() != getOutputStateSpaceSize()) {
    throw Error("Invalid size for output sample");
  }
  if (nullptr == samples) {
      samples =
          std::make_unique<gauss::gp::TrainSet>(input_sample, output_sample);
      return;
  }
  samples->addSample(input_sample, output_sample);
}

void GaussianProcessBase::pushSample(const Eigen::VectorXd &input_sample,
                                     const Eigen::VectorXd &output_sample) {
  pushSample_(input_sample, output_sample);
  updateKernel();
  updateSamplesOutputMatrix();
}

void GaussianProcessBase::clearSamples() {
  samples.reset();
  resetKernel();
  resetSamplesOutputMatrix();
}

void GaussianProcessBase::updateKernelFunction(KernelFunctionPtr new_kernel) {
  if (nullptr == new_kernel) {
    throw Error("empty kernel function");
  }
  kernelFunction = std::move(new_kernel);
  resetKernel();
  if (nullptr != samples) {
      updateKernel();
  }
}

void GaussianProcessBase::predict(const Eigen::VectorXd &point,
                                  Eigen::VectorXd &mean,
                                  double &covariance) const {
  const auto Kx_trasp = getKx(point);
  const auto Kx = Kx_trasp.transpose();
  covariance = (Kx * getKernelInverse() * Kx_trasp)(0, 0);
  covariance *= -1.0;
  covariance += kernelFunction->evaluate(point, point);
  mean = Kx * getKernelInverse() * getSamplesOutputMatrix();
}

Eigen::VectorXd GaussianProcessBase::getKx(const Eigen::VectorXd& point) const {
    const auto& input_samples = getTrainSet()->GetSamplesInput().GetSamples();
    Eigen::VectorXd Kx(input_samples.size());
    Eigen::Index pos = 0;
    for (const auto& sample : input_samples) {
        Kx(pos) = kernelFunction->evaluate(point, sample);
        ++pos;
    }
    return Kx;
}
} // namespace gauss::gp
