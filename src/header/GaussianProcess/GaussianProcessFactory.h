/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/GaussianProcessVectorial.h>
#include <GaussianUtils/components/RandomModelFactory.h>

namespace gauss::gp {
class GaussianProcessFactory
    : public RandomModelFactory<GaussianProcessVectorial>,
      public InputOutputSizeAware {
public:
  GaussianProcessFactory(const std::size_t input_space_size,
                         const std::size_t output_space_size,
                         KernelFunctionPtr kernel_function);

  std::unique_ptr<GaussianProcessVectorial> makeRandomModel() const override;

  void setSamplesNumb(const std::size_t samples) { samples_numb = samples; };

  void setInputMeanCenter(const Eigen::VectorXd &center);
  void setInputMeanScale(const Eigen::VectorXd &scale);
  void setOutputMeanCenter(const Eigen::VectorXd &center);
  void setOutputMeanScale(const Eigen::VectorXd &scale);

  std::size_t getInputStateSpaceSize() const override {
    return input_samples_center.size();
  };
  std::size_t getOutputStateSpaceSize() const override {
    return output_samples_center.size();
  };

private:
  KernelFunctionPtr kernel_function;
  std::size_t samples_numb = 500;
  Eigen::VectorXd input_samples_center;
  Eigen::VectorXd input_samples_scale;
  Eigen::VectorXd output_samples_center;
  Eigen::VectorXd output_samples_scale;
};
} // namespace gauss::gp
