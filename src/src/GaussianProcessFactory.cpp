/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/Error.h>
#include <GaussianProcess/GaussianProcessFactory.h>

namespace gauss::gp {
BoundingBoxTrainSet::BoundingBoxTrainSet(const std::size_t space_size)
    : first_corner(space_size), second_corner(space_size) {
  first_corner.setOnes();
  second_corner.setOnes();
  second_corner *= -1.0;
  updateRepresentation_();
}

void BoundingBoxTrainSet::setFirstCorner(const Eigen::VectorXd &corner) {
  if (corner.size() != first_corner.size()) {
    throw Error{"invalid corner size"};
  }
  first_corner = corner;
  updateRepresentation_();
}

void BoundingBoxTrainSet::setSecondCorner(const Eigen::VectorXd &corner) {
  if (corner.size() != second_corner.size()) {
    throw Error{"invalid corner size"};
  }
  second_corner = corner;
  updateRepresentation_();
}

void BoundingBoxTrainSet::updateRepresentation_() {
  Eigen::VectorXd min(first_corner.size()), max(first_corner.size());
  for (Eigen::Index k = 0; k < min.size(); ++k) {
    if (first_corner(k) < second_corner(k)) {
      min(k) = first_corner(k);
      max(k) = second_corner(k);
    } else {
      min(k) = second_corner(k);
      max(k) = first_corner(k);
    }
  }
  representation.traslation = min;
  representation.deltas = max - min;
}

Eigen::VectorXd BoundingBoxTrainSet::getSample() const {
  Eigen::VectorXd result;
  result.setRandom();
  result += Eigen::VectorXd::Ones(result.size());
  result /= 2.0;
  for (Eigen::Index k = 0; k < result.size(); ++k) {
    result(k) *= representation.deltas(k);
  }
  result += representation.traslation;
  return result;
}

GaussianProcessFactory::GaussianProcessFactory(
    const std::size_t input_size, const FunctionToApproximate &to_approximate,
    KernelFunctionPtr kernel_function)
    : SpaceSizesAware(input_size,
                      to_approximate(Eigen::VectorXd::Zero(input_size)).size()),
      BoundingBoxTrainSet(input_size), function_to_approximate(to_approximate) {
  if (nullptr == kernel_function) {
    throw gauss::gp::Error("empty kernel function");
  }
  this->kernel_function = std::move(kernel_function);
}

namespace {
void add_noise(Eigen::VectorXd &recipient, const GaussianDistribution &noise) {
  const auto values = noise.drawSamples(recipient.size());
  for (Eigen::Index k = 0; k < values.size(); ++k) {
    recipient(k) += values[k](0);
  }
}
} // namespace

std::unique_ptr<GaussianProcess>
GaussianProcessFactory::makeRandomModel() const {
  std::unique_ptr<GaussianProcess> result = std::make_unique<GaussianProcess>(
      kernel_function->copy(), getInputStateSpaceSize(),
      getOutputStateSpaceSize());
  for (std::size_t k = 0; k < train_set_size; ++k) {
    const auto sample_in = getSample();
    auto sample_out = function_to_approximate(sample_in);
    if (nullptr != train_set_noise) {
      add_noise(sample_out, *train_set_noise);
    }
    result->getTrainSet().addSample(sample_in, sample_out);
  }
  return result;
}

void GaussianProcessFactory::setTrainSetNoise(
    const double white_noise_standard_deviation) {
  Eigen::MatrixXd cov(1, 1);
  cov << white_noise_standard_deviation * white_noise_standard_deviation;
  train_set_noise =
      std::make_unique<GaussianDistribution>(Eigen::VectorXd::Zero(1), cov);
}
} // namespace gauss::gp
