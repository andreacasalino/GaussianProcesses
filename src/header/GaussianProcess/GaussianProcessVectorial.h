/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/components/GaussianProcessBase.h>
#include <GaussianUtils/GaussianDistribution.h>

namespace gauss::gp {
/**
 * @brief A vectorial gaussian process has an output space size greater than 1
 *
 */
class GaussianProcessVectorial : public GaussianProcessBase {
public:
  /**
   * @brief Construct a new Gaussian Process Vectorial object.
   * No initial samples would be available. Therefore, calling predict(...)
   * immediately after would throw an exception
   *
   * @param kernel
   * @param input_space_size
   * @param output_space_size
   * @throw when passing a null kernel
   * @throw when passsing input_space_size equal to 0
   * @throw when passsing output_space_size equal to 0
   */
  GaussianProcessVectorial(KernelFunctionPtr kernel,
                           const std::size_t input_space_size,
                           const std::size_t output_space_size);

  /**
   * @brief Construct a new Gaussian Process Vectorial object.
   *
   * @param kernel
   * @param train_set
   * @throw when passing a null kernel
   */
  GaussianProcessVectorial(KernelFunctionPtr kernel,
                           gauss::gp::TrainSet train_set);

  /**
   * @param point
   * @return The vectorial distribution describing the possible output of the
   * process w.r.t the passed input point.
   */
  std::vector<gauss::GaussianDistribution>
  predict(const Eigen::VectorXd &point) const;

  struct Prediction {
    Eigen::VectorXd mean;
    double covariance;
  };
  /**
   * @param point
   * @return The vectorial distribution parameters describing the possible
   * output of the process w.r.t the passed input point.
   */
  Prediction predict2(const Eigen::VectorXd &point) const;
};
} // namespace gauss::gp
