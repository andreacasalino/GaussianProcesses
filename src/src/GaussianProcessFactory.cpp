/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/GaussianProcessFactory.h>

namespace gauss::gp {
GaussianProcessFactory::GaussianProcessFactory(
    const std::size_t input_space_size, const std::size_t output_space_size,
    KernelFunctionPtr kernel_function)
    : input_samples_center(input_space_size),
      input_samples_scale(input_space_size),
      output_samples_center(output_space_size),
      output_samples_scale(output_space_size) {
  if (nullptr == kernel_function) {
    throw gauss::gp::Error("empty kernel function");
  }
  this->kernel_function = std::move(kernel_function);
  if ((0 == input_space_size) || (0 == output_space_size)) {
    throw gauss::gp::Error("Invalid space size");
  }
  input_samples_center.setZero();
  input_samples_scale.setOnes();
  output_samples_center.setZero();
  output_samples_scale.setOnes();
}

namespace {
std::vector<Eigen::VectorXd> get_samples(const Eigen::VectorXd &center,
                                         const Eigen::VectorXd &scale,
                                         const std::size_t samples) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples);
  Eigen::VectorXd delta(scale.size());
  for (std::size_t k = 0; k < samples; ++k) {
    result.emplace_back(center);
    delta.setRandom();
    for (Eigen::Index i = 0; i < delta.size(); ++i) {
      delta(i) *= scale(i);
    }
    result.back() += delta;
  }
  return result;
}
} // namespace

std::unique_ptr<GaussianProcessVectorial>
GaussianProcessFactory::makeRandomModel() const {
  gauss::gp::TrainSet samples{
      get_samples(input_samples_center, input_samples_scale, samples_numb),
      get_samples(output_samples_center, output_samples_scale, samples_numb)};

  return std::make_unique<GaussianProcessVectorial>(kernel_function->copy(),
                                                    std::move(samples));
};

namespace {
void check_and_assign(Eigen::VectorXd &recipient,
                      const Eigen::VectorXd &new_val) {
  if (new_val.size() != recipient.size()) {
    throw gauss::gp::Error("invalid size");
  }
  recipient = new_val;
}
} // namespace

void GaussianProcessFactory::setInputMeanCenter(const Eigen::VectorXd &center) {
  check_and_assign(input_samples_center, center);
};
void GaussianProcessFactory::setInputMeanScale(const Eigen::VectorXd &scale) {
  check_and_assign(input_samples_scale, scale);
};

void GaussianProcessFactory::setOutputMeanCenter(
    const Eigen::VectorXd &center) {
  check_and_assign(output_samples_center, center);
};
void GaussianProcessFactory::setOutputMeanScale(const Eigen::VectorXd &scale) {
  check_and_assign(output_samples_scale, scale);
};
} // namespace gauss::gp
