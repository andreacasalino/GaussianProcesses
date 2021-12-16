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
      public InputOutputSizeAwareBase {
public:
    GaussianProcessFactory(const std::size_t input_space_size,
                              const std::size_t output_space_size);

    GaussianProcessFactory(const std::size_t input_space_size,
                              const std::size_t output_space_size,
                              KernelFunctionPtr kernel_function);

  std::unique_ptr<GaussianProcessVectorial> makeRandomModel() const override;

  void setInputMeanCenter(const Eigen::VectorXd &center);
  void setInputMeanScale(const Eigen::VectorXd &scale);
  void setOutputMeanCenter(const Eigen::VectorXd &center);
  void setOutputMeanScale(const Eigen::VectorXd &scale);

private:
  KernelFunctionPtr kernel_function;
  Eigen::VectorXd input_samples_center;
  Eigen::VectorXd input_samples_scale;
  Eigen::VectorXd output_samples_center;
  Eigen::VectorXd output_samples_scale;
};
} // namespace gauss::gp
