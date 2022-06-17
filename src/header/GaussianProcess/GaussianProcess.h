/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/KernelCovariance.h>
#include <GaussianProcess/YYMatrices.h>

#include <TrainingTools/Trainer.h>

#include <GaussianUtils/GaussianDistribution.h>

#include <optional>

namespace gauss::gp {
class GaussianProcess : public KernelCovariance,
                        public YYMatrixTrain,
                        public YYMatrixPredict {
public:
  template <typename... TrainSetArgs>
  GaussianProcess(KernelFunctionPtr kernel, TrainSetArgs... args)
      : KernelCovariance(std::move(kernel)),
        samples(std::forward<TrainSetArgs>(args)...) {}

  const TrainSet &getTrainSet() const final { return samples; }
  TrainSet &getTrainSet() { return samples; }

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
  Eigen::VectorXd getHyperParametersGradient() const;

  /**
   * @param point
   * @return The vectorial distribution describing the possible output of the
   * process w.r.t the passed input point.
   */
  std::vector<gauss::GaussianDistribution>
  predict(const Eigen::VectorXd &point,
          const bool accept_bad_covariance = true) const;

  struct Prediction {
    Eigen::VectorXd mean;
    double covariance;
  };
  /**
   * @param point
   * @return The vectorial distribution parameters describing the possible
   * output of the process w.r.t the passed input point.
   */
  Prediction predict2(const Eigen::VectorXd &point,
                      const bool accept_bad_covariance = true) const;

  GaussianDistribution predict3(const Eigen::VectorXd &point,
                                const bool accept_bad_covariance = true) const;

protected:
  Eigen::VectorXd predict(const Eigen::VectorXd &point, double &covariance,
                          const bool accept_bad_covariance) const;

private:
  TrainSet samples;
};

template <std::size_t InputSize, std::size_t OutputSize>
class GaussianProcessVectorial : public GaussianProcess {
public:
  GaussianProcessVectorial(KernelFunctionPtr kernel)
      : GaussianProcess(std::move(kernel), InputSize, OutputSize) {}
};

template <std::size_t InputSize>
using GaussianProcessScalar = GaussianProcessVectorial<InputSize, 1>;

void train(GaussianProcess &subject, ::train::Trainer &trainer,
           const std::optional<gauss::GaussianDistribution>
               &hyperparameters_prior = std::nullopt);
} // namespace gauss::gp
