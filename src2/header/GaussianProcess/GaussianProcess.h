/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/Error.h>
#include <GaussianProcess/GaussianProcessBase.h>
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
  template <typename... TrainSetArgs>
  GaussianProcess(KernelFunctionPtr kernel, TrainSetArgs... args)
      : GaussianProcessBase(std::move(kernel),
                            std::forward<TrainSetArgs>(args)...) {
    if (1 != getTrainSet().getOutputStateSpaceSize()) {
      throw Error{"Invalid size of output space"};
    }
  }

  /**
   * @param point
   * @return The distribution describing the possible output of the
   * process w.r.t the passed input point.
   */
  gauss::GaussianDistribution predict(const Eigen::VectorXd &point) const;

  struct Prediction {
    double mean;
    double covariance;
  };
  /**
   * @param point
   * @return The parameters distribution describing the possible output of the
   * process w.r.t the passed input point.
   */
  Prediction predict2(const Eigen::VectorXd &point) const;
};
} // namespace gauss::gp
