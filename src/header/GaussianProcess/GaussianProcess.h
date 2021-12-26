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
 * @brief A normal gaussian process has an output space size w equal to 1
 *
 */
class GaussianProcess : public GaussianProcessBase {
public:
  /**
   * @brief Construct a new Gaussian Process Vectorial object.
   * No initial samples would be available. Therefore, calling predict(...)
   * immediately after would throw an exception
   *
   * @param kernel
   * @param input_space_size
   * @throw when passing a null kernel
   * @throw when passsing input_space_size equal to 0
   */
  GaussianProcess(KernelFunctionPtr kernel, const std::size_t input_space_size);

  /**
   * @brief Construct a new Gaussian Process Vectorial object.
   *
   * @param kernel
   * @param train_set
   * @throw when passing a null kernel
   * @throw passing a train set with output samples with a size different from 1
   */
  GaussianProcess(KernelFunctionPtr kernel, gauss::gp::TrainSet train_set);

  /**
   * @param point
   * @return The vectorial distribution describing the possible output of the
   * process w.r.t the passed input point.
   */
  gauss::GaussianDistribution predict(const Eigen::VectorXd &point) const;
};
} // namespace gauss::gp
