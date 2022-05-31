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
    : SizesAwareBase(input_space_size, output_space_size),
      KernelMatrix(std::move(kernel)) {}

GaussianProcessBase::GaussianProcessBase(KernelFunctionPtr kernel,
                                         gauss::gp::TrainSet train_set)
    : GaussianProcessBase(std::move(kernel), train_set.getInputStateSpaceSize(),
                          train_set.getOutputStateSpaceSize()) {
  this->samples = std::make_unique<gauss::gp::TrainSet>(std::move(train_set));
  updateKernelMatrix();
  updateOutputMatrix();
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
  updateKernelMatrix();
  updateOutputMatrix();
}

void GaussianProcessBase::clearSamples() {
  samples.reset();
  resetKernelMatrix();
  resetOutputMatrix();
}

void GaussianProcessBase::updateKernelFunction(KernelFunctionPtr new_kernel) {
  if (nullptr == new_kernel) {
    throw Error("empty kernel function");
  }
  this->KernelMatrix::updateKernelFuction(std::move(new_kernel));
  if (nullptr != samples) {
    updateKernelMatrix();
  }
}

Eigen::VectorXd GaussianProcessBase::predict(const Eigen::VectorXd &point,
                                             double &covariance) const {
  Eigen::VectorXd Kx = getKx(point);
  const Eigen::MatrixXd &K_inverse = getKernelMatrixInverse();
  covariance = getKernelFunction().evaluate(point, point);
  covariance -= Kx.dot(K_inverse * Kx);
  covariance = abs(covariance);
  Eigen::MatrixXd K_inverse_Output = K_inverse * getOutputMatrix();
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
  const auto &kernel_function = getKernelFunction();
  for (const auto &sample : input_samples) {
    Kx(pos) = kernel_function.evaluate(point, sample);
    ++pos;
  }
  return Kx;
}

Eigen::VectorXd GaussianProcessBase::getHyperParameters() const {
  const auto &params = getKernelFunction().getParameters();
  Eigen::VectorXd result(params.size());
  Eigen::Index i = 0;
  for (const auto &parameter : params) {
    result(i) = parameter;
    ++i;
  }
  return result;
}

void GaussianProcessBase::setHyperParameters(
    const Eigen::VectorXd &parameters) {
  auto &kernel_function = getKernelFunction();
  if (static_cast<std::size_t>(parameters.size()) !=
      kernel_function.numberOfParameters()) {
    throw gauss::gp::Error("Invalid new set of parameters");
  }
  Eigen::Index i = 0;
  std::vector<double> params_vec;
  params_vec.reserve(static_cast<std::size_t>(parameters.size()));
  for (auto &parameter : parameters) {
    params_vec.push_back(parameters(i));
    ++i;
  }
  resetKernelMatrix();
  if (nullptr != samples) {
    updateKernelMatrix();
  }
}

double GaussianProcessBase::getLogLikelihood() const {
  double result = 0.0;
  result -= 0.5 *
            static_cast<double>(
                samples->GetSamplesInput().GetSamples().front().size()) *
            log(getCovarianceDeterminant());
  result -= 0.5 * (getOutputMatrix() * getCovarianceInv()).trace();
  return result;
}

Eigen::VectorXd GaussianProcessBase::getParametersGradient() const {
  const auto parameters_numb = getKernelFunction().numberOfParameters();
  const auto &samples = getTrainSet()->GetSamplesInput().GetSamples();

  std::vector<Eigen::MatrixXd> kernel_matrix_gradients;
  {
    kernel_matrix_gradients.reserve(parameters_numb);
    for (std::size_t p = 0; p < parameters_numb; ++p) {
      kernel_matrix_gradients.emplace_back(
          static_cast<Eigen::Index>(samples.size()),
          static_cast<Eigen::Index>(samples.size()));
    }
    auto fill_kernel_matrix_gradients_at =
        [&kernel_function = getKernelFunction(), &samples,
         &kernel_matrix_gradients](const std::size_t r, const std::size_t c) {
          auto grad = kernel_function.getGradient(samples[r], samples[c]);
          for (std::size_t k = 0; k < grad.size(); ++k) {
            kernel_matrix_gradients[k](r, c) = grad[k];
            if (r != c) {
              kernel_matrix_gradients[k](c, r) =
                  kernel_matrix_gradients[k](r, c);
            }
          }
        };
    for (std::size_t r = 0; r < samples.size(); ++r) {
      for (std::size_t c = r; c < samples.size(); ++c) {
        fill_kernel_matrix_gradients_at(r, c);
      }
    }
  }

  Eigen::MatrixXd YY = getOutputMatrix();
  Eigen::VectorXd result(static_cast<Eigen::Index>(parameters_numb));
  const Eigen::MatrixXd &kernel_inv = getKernelMatrixInverse();
  double M = static_cast<double>(samples.front().size());
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(parameters_numb);
       ++i) {
    const auto &gradient_matrix =
        kernel_matrix_gradients[static_cast<std::size_t>(i)];
    result(i) = -M * (kernel_inv * gradient_matrix).trace();
    result(i) += (YY * kernel_inv * gradient_matrix * kernel_inv).trace();
    result(i) *= 0.5;
  }
  return result;
};

void GaussianProcessBase::train(::train::Trainer &trainer) {
  trainer.maximize();
  trainer.train(*this);
}
} // namespace gauss::gp
