/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/Error.h>
#include <GaussianProcess/components/KernelMatrix.h>
#include <GaussianProcess/components/OutputMatrix.h>
#include <GaussianProcess/components/SizesAware.h>

#include <TrainingTools/ParametersAware.h>
#include <TrainingTools/Trainer.h>

namespace gauss::gp {
class GaussianProcessBase : public SizesAwareBase,
                            public KernelMatrix,
                            public OutputMatrix,
                            protected ::train::ParametersAware {
public:
  TrainSet &getSamples();
  const TrainSet &getSamples() const;

  /// column wise
  Eigen::VectorXd getKx(const Eigen::VectorXd &point) const;

  /**
   * @return the tunable parameters of the kernel function
   */
  Eigen::VectorXd getHyperParameters() const;
  /**
   * @param parameters , the new set of tunable parameters for the kernel
   * function
   * @throw in case the number of parameters is not consistent
   */
  void setHyperParameters(const Eigen::VectorXd &parameters);

  /**
   * @return the logarithmic likelihood of the process, i.e. the product of the
   * logarithmic likelihoods of the element iniside the samples used to build
   * the kernel, w.r.t the process itself
   */
  double getLogLikelihood() const;

  /**
   * @return The gradient of the tunable parameters w.r.t. to the logarithmic
   * likelihood
   */
  Eigen::VectorXd getParametersGradient() const;

  void train(::train::Trainer &trainer);

protected:
  GaussianProcessBase(KernelFunctionPtr kernel, ...);

  Eigen::VectorXd predict(const Eigen::VectorXd &point,
                          double &covariance) const;

  ::train::Vect getParameters() const final { return getHyperParameters(); }
  void setParameters(const ::train::Vect &parameters) final {
    setHyperParameters(parameters);
  }
  ::train::Vect getGradient() const final { return getParametersGradient(); };

private:
  gauss::gp::TrainSet samples;
};
} // namespace gauss::gp
