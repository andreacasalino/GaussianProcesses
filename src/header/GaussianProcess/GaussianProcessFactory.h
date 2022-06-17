/**
 * Author:    Andrea Casalino
 * Created:   26.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianUtils/components/RandomModelFactory.h>

#include <functional>

namespace gauss::gp {
/**
 * @brief If tou never call setFirstCorner or setSecondCorner, corners are
 * default assumed equal to (-1, ..., -1) and (1, ..., 1)
 *
 */
class BoundingBoxTrainSet {
public:
  void setFirstCorner(const Eigen::VectorXd &corner);
  void setSecondCorner(const Eigen::VectorXd &corner);

protected:
  BoundingBoxTrainSet(const std::size_t space_size);

  Eigen::VectorXd getSample() const;

  void updateRepresentation_();

private:
  Eigen::VectorXd first_corner;
  Eigen::VectorXd second_corner;

  struct BoxRepresentation {
    Eigen::VectorXd deltas;
    Eigen::VectorXd traslation;
  };
  BoxRepresentation representation;
};

using FunctionToApproximate =
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)>;

/**
 * @brief Builds a GP, creating the training set by sampling points in an hyper
 * bounding box region.
 */
class GaussianProcessFactory : public RandomModelFactory<GaussianProcess>,
                               public SpaceSizesAware,
                               public BoundingBoxTrainSet {
public:
  GaussianProcessFactory(const std::size_t input_size,
                         const FunctionToApproximate &to_approximate,
                         KernelFunctionPtr kernel_function);

  std::unique_ptr<GaussianProcess> makeRandomModel() const override;

  void setTrainSetSize(const std::size_t size) { train_set_size = size; };
  void setTrainSetNoise(const double white_noise_standard_deviation);

private:
  const FunctionToApproximate function_to_approximate;

  KernelFunctionPtr kernel_function;

  std::size_t train_set_size = 100;

  std::unique_ptr<GaussianDistribution> train_set_noise;
};
} // namespace gauss::gp
