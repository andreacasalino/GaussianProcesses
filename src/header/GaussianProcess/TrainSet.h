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

  TrainSet(const std::string &file_to_read, const std::size_t input_space_size);

  template <typename CollectionInput, typename CollectionOutput>
  TrainSet(const CollectionInput &input_samples,
           const CollectionOutput &output_samples)
      : input(input_samples), output(output_samples){};

  TrainSet(const Eigen::VectorXd &initial_input_sample,
           const Eigen::VectorXd &initial_output_sample)
      : input(std::vector<Eigen::VectorXd>{initial_input_sample}),
        output(std::vector<Eigen::VectorXd>{initial_output_sample}){};

  void operator+=(const gauss::gp::TrainSet &o) {
    input += o.GetSamplesInput();
    output += o.GetSamplesOutput();
  };

  void addSample(const Eigen::VectorXd &input_sample,
                 const Eigen::VectorXd &output_sample) {
    input.addSample(input_sample);
    output.addSample(output_sample);
  };

  const gauss::TrainSet &GetSamplesInput() const { return input; };
  const gauss::TrainSet &GetSamplesOutput() const { return output; };

private:
  gauss::TrainSet input;
  gauss::TrainSet output;
};
} // namespace gauss::gp
