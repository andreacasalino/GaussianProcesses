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
class NegativeCovariancePredictionError : public Error {
public:
  NegativeCovariancePredictionError();

  static const double TOLLERANCE;
};

struct RawValuesPredictionTag {};
static constexpr RawValuesPredictionTag RAW_VALUES_PREDICTION_TAG =
    RawValuesPredictionTag{};

struct SinglePredictiveDistributionTag {};
static constexpr SinglePredictiveDistributionTag
    SINGLE_PREDICTIVE_DISTRIBUTION_TAG = SinglePredictiveDistributionTag{};

class GaussianProcess : public KernelCovariance,
                        public YYMatrixTrain,
                        public YYMatrixPredict {
public:
  /**
   * @param kernel the kernel function to absorb
   * @param args any set of parameters that is valid to build an initilal
   * gauss::gp::TrainSet
   */
  template <typename... TrainSetArgs>
  GaussianProcess(KernelFunctionPtr kernel, TrainSetArgs... args)
      : KernelCovariance(std::move(kernel)),
        samples(std::forward<TrainSetArgs>(args)...) {}

  const TrainSet &getTrainSet() const final { return samples; }
  TrainSet &getTrainSet() { return samples; }

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
   * @return the logarithmic likelihood of the gausian process, i.e. the
   * summation of the logarithmic likelihoods of the element iniside the samples
   * used to build the kernel, w.r.t the process itself. Refer also to the pdf
   * documentation.
   */
  double getLogLikelihood() const;

  /**
   * @return The gradient of the tunable parameters of the kernel function.
   */
  Eigen::VectorXd getHyperParametersGradient() const;

  /**
   * @param point the input whose output should be predicted
   * @return a vector storing a normal distribution for each component of the
   * predicted output. Notice that all the returned distributions have the same
   * covariance value (refer also to the pdf documentation to understand why).
   * @param accept_bad_covariance when true, too low or negative value for the
   * prediction covariance are accepted. Otherwise a
   * NegativeCovariancePredictionError is thrown in case of getting bad values
   * for the covariance.
   */
  std::vector<gauss::GaussianDistribution>
  predict(const Eigen::VectorXd &point,
          const bool accept_bad_covariance = true) const;

  struct Prediction {
    Eigen::VectorXd mean;
    double covariance;
  };
  /**
   * @brief similar to GaussianProcess::predict, but returning the prediction
   * into a data structure storing the mean and the covariance.
   */
  Prediction
  predict(const Eigen::VectorXd &point,
          const RawValuesPredictionTag &tag = RAW_VALUES_PREDICTION_TAG,
          const bool accept_bad_covariance = true) const;

  /**
   * @brief similar to GaussianProcess::predict, but returning the prediction
   * into a single multivariate Gaussian distribution
   */
  GaussianDistribution predict(const Eigen::VectorXd &point,
                               const SinglePredictiveDistributionTag &tag =
                                   SINGLE_PREDICTIVE_DISTRIBUTION_TAG,
                               const bool accept_bad_covariance = true) const;

protected:
  Eigen::VectorXd predict_(const Eigen::VectorXd &point, double &covariance,
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

/**
 * @brief tune the hyperparameters of the process by learning
 *
 * @param subject
 * @param trainer
 * @param hyperparameters_prior when specified, represents a multivariate prior
 * gaussian distribution (refer also to the pdf documentation) over the
 * hyperparamters which is accounted for computing the gradient (and actually
 * find the optimum point).
 * @throw in case the size of hyperparameters_prior (if specified) is
 * inconsistent with the number of hyperparameters of the process
 */
void train(GaussianProcess &subject, ::train::Trainer &trainer,
           const std::optional<gauss::GaussianDistribution>
               &hyperparameters_prior = std::nullopt);
} // namespace gauss::gp
