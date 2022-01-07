/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/Error.h>
#include <GaussianProcess/components/InputOutputSizeAware.h>
#include <GaussianProcess/components/KernelAware.h>
#include <GaussianProcess/components/OutputSetMatrixAware.h>

#include <TrainingTools/ParametersAware.h>
#include <TrainingTools/Trainer.h>

namespace gauss::gp {
class GaussianProcessBase : public InputOutputSizeAwareBase,
                            public KernelAware,
                            public OutputSetMatrixAware,
                            public ::train::ParametersAware {
public:
  /**
   * @brief replace the kernel function.
   *
   * @param new_kernel
   * @throw passing a null kernel
   */
  void updateKernelFunction(KernelFunctionPtr new_kernel);

  /**
   * @brief Add one new pair of samples to enlarge the kernel
   *
   * @param input_sample
   * @param output_sample
   * @throw in case the size of the samples is not consistent with the input or
   * the output space size of the process
   */
  void pushSample(const Eigen::VectorXd &input_sample,
                  const Eigen::VectorXd &output_sample);
  /**
   * @brief Add a new set of input-output realizations to enlarge the kernel
   *
   * @tparam IterableT
   * @param input_samples
   * @param output_samples
   * @throw in case the size of the samples is not consistent with the
   * input or the output space size of the process
   */
  template <typename IterableT>
  void pushSamples(const IterableT &input_samples,
                   const IterableT &output_samples) {
    if (input_samples.size() != output_samples.size()) {
      throw Error("Invalid new sets of samples");
    }
    auto it_out = output_samples.begin();
    for (const auto &sample_in : input_samples) {
      pushSample_(sample_in, *it_out);
      ++it_out;
    }
    updateKernel();
    updateSamplesOutputMatrix();
  };
  /**
   * @brief Remove all the samples.
   * This will empty the kernel and will make a subsequent call to predict(...)
   * throw.
   *
   */
  void clearSamples();

  /// column wise
  Eigen::VectorXd getKx(const Eigen::VectorXd &point) const;

  /**
   * @return the samples used to build the kernel
   */
  const TrainSet *getTrainSet() const override { return samples.get(); };

  /**
   * @return the tunable parameters of the kernel function
   */
  Eigen::VectorXd getParameters() const override;
  /**
   * @param parameters , the new set of tunable parameters for the kernel
   * function
   * @throw in case the number of parameters is not consistent
   */
  void setParameters(const Eigen::VectorXd &parameters) override;

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
  ::train::Vect getGradient() const override {
    return getParametersGradient();
  };

  void train(::train::Trainer &trainer);

protected:
  GaussianProcessBase(KernelFunctionPtr kernel,
                      const std::size_t input_space_size,
                      const std::size_t output_space_size);

  GaussianProcessBase(KernelFunctionPtr kernel, gauss::gp::TrainSet train_set);

  Eigen::VectorXd predict(const Eigen::VectorXd &point,
                          double &covariance) const;

private:
  void pushSample_(const Eigen::VectorXd &input_sample,
                   const Eigen::VectorXd &output_sample);

  std::unique_ptr<gauss::gp::TrainSet> samples;
  std::vector<ParameterHandlerPtr> parameters;
};
} // namespace gauss::gp
