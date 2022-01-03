/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/components/GaussianProcessBase.h>

#include <iostream>

namespace gauss::gp {
GaussianProcessBase::GaussianProcessBase(KernelFunctionPtr kernel,
                                         const std::size_t input_space_size,
                                         const std::size_t output_space_size)
    : InputOutputSizeAwareBase(input_space_size, output_space_size),
      KernelAware(std::move(kernel)) {
  parameters = kernelFunction->getParameters();
}

GaussianProcessBase::GaussianProcessBase(KernelFunctionPtr kernel,
                                         gauss::gp::TrainSet train_set)
    : GaussianProcessBase(std::move(kernel), train_set.getInputStateSpaceSize(),
                          train_set.getOutputStateSpaceSize()) {
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
  parameters = kernelFunction->getParameters();
  resetKernel();
  if (nullptr != samples) {
    updateKernel();
  }
}

Eigen::VectorXd GaussianProcessBase::predict(const Eigen::VectorXd &point,
                                             double &covariance) const {
  Eigen::VectorXd Kx = getKx(point);
  const Eigen::MatrixXd &K_inverse = kernel->getKernelInv();
  covariance = kernelFunction->evaluate(point, point);
  covariance -= Kx.dot(K_inverse * Kx);
  covariance = abs(covariance);
  Eigen::MatrixXd K_inverse_Output = K_inverse * getSamplesOutputMatrix();
  Eigen::VectorXd result(K_inverse_Output.cols());
  for (Eigen::Index i = 0; i < K_inverse_Output.cols(); ++i) {
    result(i) = Kx.dot(K_inverse_Output.col(i));
  }
  return result;
}

Eigen::VectorXd GaussianProcessBase::getKx(const Eigen::VectorXd &point) const {
  const auto &input_samples = getTrainSet()->GetSamplesInput().GetSamples();
  Eigen::VectorXd Kx(input_samples.size());
  Eigen::Index pos = 0;
  for (const auto &sample : input_samples) {
    Kx(pos) = kernelFunction->evaluate(point, sample);
    ++pos;
  }
  return Kx;
}

Eigen::VectorXd GaussianProcessBase::getParameters() const {
  Eigen::VectorXd result(parameters.size());
  Eigen::Index i = 0;
  for (const auto &parameter : this->parameters) {
    result(i) = parameter->getParameter();
    ++i;
  }
  return result;
}

void GaussianProcessBase::setParameters(const Eigen::VectorXd &parameters) {
  if (parameters.size() != this->parameters.size()) {
    throw gauss::gp::Error("Invalid new set of parameters");
  }
  Eigen::Index i = 0;
  for (auto &parameter : this->parameters) {
    parameter->setParameter(parameters(i));
    ++i;
  }
  resetKernel();
  if (nullptr != samples) {
    updateKernel();
  }
};

double GaussianProcessBase::getLikelihood() const {
  Eigen::MatrixXd Y_Ytras =
      getSamplesOutputMatrix() * getSamplesOutputMatrix().transpose();
  double result = 0.0;
  result -= 0.5 * samples->GetSamplesInput().GetSamples().size() *
            log(getCovarianceDeterminant());
  result -= 0.5 * (getCovarianceInv() * Y_Ytras).trace();
  return result;
}

namespace {
Eigen::MatrixXd
compute_kernel_gradient(const gauss::gp::ParameterHandler &handler,
                        const std::vector<Eigen::VectorXd> &samples) {
  Eigen::MatrixXd result(samples.size(), samples.size());
  Eigen::Index row = 0, col;
  for (auto it = samples.begin(); it != samples.end(); ++it, ++row) {
    col = row;
    for (auto it2 = it; it2 != samples.end(); ++it2, ++col) {
      result(row, col) = handler.evaluate_gradient(*it, *it2);
      result(col, row) = result(row, col);
    }
  }
  return result;
}
} // namespace

Eigen::VectorXd GaussianProcessBase::getParametersGradient() const {
  Eigen::MatrixXd Y_Ytras =
      getSamplesOutputMatrix() * getSamplesOutputMatrix().transpose();

  Eigen::VectorXd result(parameters.size());
  Eigen::Index i = 0;
  const Eigen::MatrixXd &kernel_inv = kernel->getKernelInv();
  const auto &samples_input = samples->GetSamplesInput().GetSamples();
  for (auto &parameter : this->parameters) {
    Eigen::MatrixXd kernel_gradient =
        compute_kernel_gradient(*parameter, samples_input);
    result(i) = -samples->GetSamplesInput().GetSamples().size() *
                (kernel_inv * kernel_gradient).trace();
    result(i) += (kernel_inv * kernel_gradient * kernel_inv * Y_Ytras).trace();
    result(i) *= 0.5;
    ++i;
  }
  std::cout << result.transpose() << std::endl;
  return -result;
};
} // namespace gauss::gp
