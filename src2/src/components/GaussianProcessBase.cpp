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
  kernelFunction = std::move(new_kernel);
  resetKernelMatrix();
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

Eigen::VectorXd GaussianProcessBase::getParameters() const {
  const auto &params = getKernelFunction().getParameters();
  Eigen::VectorXd result(params.size());
  Eigen::Index i = 0;
  for (const auto &parameter : params) {
    result(i) = parameter;
    ++i;
  }
  return result;
}

void GaussianProcessBase::setParameters(const Eigen::VectorXd &parameters) {
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
};

namespace {
Eigen::MatrixXd compute_V_Vtrasp(const Eigen::VectorXd &v) {
  Eigen::MatrixXd result(v.size(), v.size());
  for (Eigen::Index r = 0; r < v.size(); ++r) {
    for (Eigen::Index c = 0; c < v.size(); ++c) {
      result(r, c) = v(r) * v(c);
    }
  }
  return result;
}

Eigen::MatrixXd
compute_kernel_gradient(const gauss::gp::ParameterHandler &handler,
                        const std::vector<Eigen::VectorXd> &samples) {
  Eigen::MatrixXd result(samples.size(), samples.size());
  Eigen::Index row = 0, col;
  for (auto it = samples.begin(); it != samples.end(); ++it, ++row) {
    col = row;
    for (auto it2 = it; it2 != samples.end(); ++it2, ++col) {
      result(row, col) = handler.evaluate_gradient(*it, *it2);
      if (col != row) {
        result(col, row) = result(row, col);
      }
    }
  }
  return result;
}
} // namespace

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
  Eigen::MatrixXd Y_Y = getOutputMatrix();

  Eigen::VectorXd result(static_cast<Eigen::Index>(parameters.size()));
  const Eigen::MatrixXd &kernel_inv = kernel->getKernelInv();
  const auto &samples_input = samples->GetSamplesInput().GetSamples();
  double M = static_cast<double>(samples_input.front().size());
  Eigen::Index i = 0;
  for (const auto &parameter : this->parameters) {
    Eigen::MatrixXd kernel_gradient =
        compute_kernel_gradient(*parameter, samples_input);
    result(i) = -M * (kernel_inv * kernel_gradient).trace();
    result(i) += (Y_Y * kernel_inv * kernel_gradient * kernel_inv).trace();
    result(i) *= 0.5;
    ++i;
  }
  return -result;
};

void GaussianProcessBase::train(::train::Trainer &trainer) {
  trainer.maximize();
  trainer.train(*this);
}
} // namespace gauss::gp
