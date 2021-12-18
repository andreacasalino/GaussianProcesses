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

namespace gauss::gp {
class GaussianProcessBase : public InputOutputSizeAwareBase,
                            public KernelAware,
                            public OutputSetMatrixAware {
public:
  void updateKernelFunction(KernelFunctionPtr new_kernel);

  void pushSample(const Eigen::VectorXd &input_sample,
                  const Eigen::VectorXd &output_sample);
  template <typename IterableT>
  void pushSample(const IterableT &input_samples,
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
  void clearSamples();

  // row wise
  Eigen::VectorXd getKx(const Eigen::VectorXd &point) const;

  const TrainSet *getTrainSet() const override { return samples.get(); };

  Eigen::VectorXd getParameters() const;
  void setParameters(const Eigen::VectorXd &parameters);

  double getLikelihood() const;
  Eigen::VectorXd getParametersGradient() const;

protected:
  GaussianProcessBase(const GaussianProcessBase &);
  GaussianProcessBase &operator=(const GaussianProcessBase &);

  GaussianProcessBase(GaussianProcessBase &&);
  GaussianProcessBase &operator=(GaussianProcessBase &&);

  GaussianProcessBase(KernelFunctionPtr kernel,
                      const std::size_t input_space_size,
                      const std::size_t output_space_size);

  GaussianProcessBase(KernelFunctionPtr kernel, gauss::gp::TrainSet train_set);

  void predict(const Eigen::VectorXd &point, Eigen::VectorXd &mean,
               double &covariance) const;

private:
  void pushSample_(const Eigen::VectorXd &input_sample,
                   const Eigen::VectorXd &output_sample);

  std::unique_ptr<gauss::gp::TrainSet> samples;
  std::vector<ParameterHandlerPtr> parameters;
};
} // namespace gauss::gp
