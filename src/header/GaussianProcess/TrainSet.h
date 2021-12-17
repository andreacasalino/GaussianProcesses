/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/components/InputOutputSizeAware.h>
#include <GaussianUtils/TrainSet.h>
#include <memory>

namespace gauss::gp {
class TrainSet : public InputOutputSizeAware {
public:
  TrainSet(const TrainSet &o) = default;
  TrainSet &operator=(const TrainSet &o) = default;

  TrainSet(TrainSet &&o) = default;
  TrainSet &operator=(TrainSet &&o) = default;

  TrainSet(gauss::TrainSet input_samples, gauss::TrainSet output_samples);

  TrainSet(const std::string &file_to_read, const std::size_t input_space_size);

  template <typename CollectionInput, typename CollectionOutput>
  TrainSet(const CollectionInput& input_samples,
      const CollectionOutput& output_samples) {
      input = std::make_unique<gauss::TrainSet>(input_samples);
      output = std::make_unique<gauss::TrainSet>(output_samples);
  };

  TrainSet(const Eigen::VectorXd& initial_input_sample,
      const Eigen::VectorXd& initial_output_sample) {
      input = std::make_unique<gauss::TrainSet>(initial_input_sample);
      output = std::make_unique<gauss::TrainSet>(initial_output_sample);
  };

  void operator+=(const gauss::gp::TrainSet &o) {
    *input += o.GetSamplesInput();
    *output += o.GetSamplesOutput();
  };

  void operator+=(const Eigen::VectorXd& sample);

  void addSample(const Eigen::VectorXd &input_sample,
                 const Eigen::VectorXd &output_sample) {
    *input += input_sample;
    *output += output_sample;
  };

  std::size_t getInputStateSpaceSize() const override { return input->GetSamples().front().size(); };
  std::size_t getOutputStateSpaceSize() const override { return output->GetSamples().front().size(); };

  const gauss::TrainSet &GetSamplesInput() const { return *input; };
  const gauss::TrainSet &GetSamplesOutput() const { return *output; };

private:
  std::unique_ptr<gauss::TrainSet> input;
  std::unique_ptr<gauss::TrainSet> output;
};
} // namespace gauss::gp
