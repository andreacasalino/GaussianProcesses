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

  /**
   * @brief Construct a new Train Set object.
   * For each realization inside input_samples corresponds a
   * realization in output_samples. Indeed, the two inputs describe pairs of
   * possible inputs-outputs
   *
   * @param input_samples
   * @param output_samples
   * @throw In case the 2 passed sets have a different size
   */
  TrainSet(gauss::TrainSet input_samples, gauss::TrainSet output_samples);

  /**
   * @brief Construct a new Train Set object.
   * input_samples and output_samples are red from file.
   * In each row of such a file, the initial columns represent the inpuy
   * realization, while the remaining the output one. The size of th input
   * samples are assumed equal to input_space_size
   *
   * @param file_to_read
   * @param input_space_size
   * @throw In case the file is not existent
   * @throw In case the file contains less columns than input_space_size
   */
  TrainSet(const std::string &file_to_read, const std::size_t input_space_size);

  /**
   * @brief Construct a new Train Set object.
   * Similar to TrainSet(gauss::TrainSet , gauss::TrainSet), but passing two
   * iterable sets
   *
   * @tparam CollectionInput
   * @tparam CollectionOutput
   * @param input_samples, an iterable collections of input realizations
   * @param output_samples, an iterable collections of output realizations
   */
  template <typename CollectionInput, typename CollectionOutput>
  TrainSet(const CollectionInput &input_samples,
           const CollectionOutput &output_samples) {
    input = std::make_unique<gauss::TrainSet>(input_samples);
    output = std::make_unique<gauss::TrainSet>(output_samples);
  };

  /**
   * @brief Construct a new Train Set object, assuming a single input-output
   * realization pair
   *
   * @param initial_input_sample
   * @param initial_output_sample
   */
  TrainSet(const Eigen::VectorXd &initial_input_sample,
           const Eigen::VectorXd &initial_output_sample) {
    input = std::make_unique<gauss::TrainSet>(initial_input_sample);
    output = std::make_unique<gauss::TrainSet>(initial_output_sample);
  };

  void operator+=(const gauss::gp::TrainSet &o) {
    *input += o.GetSamplesInput();
    *output += o.GetSamplesOutput();
  };

  /**
   * @brief Add a new pair input-output sample.
   * The values of the input part is assumed as the initial values of the passed
   * vector, while the remaining values are assumed to make the output part.
   *
   * @param sample
   */
  void operator+=(const Eigen::VectorXd &sample);

  void addSample(const Eigen::VectorXd &input_sample,
                 const Eigen::VectorXd &output_sample) {
    *input += input_sample;
    *output += output_sample;
  };

  std::size_t getInputStateSpaceSize() const override {
    return input->GetSamples().front().size();
  };
  std::size_t getOutputStateSpaceSize() const override {
    return output->GetSamples().front().size();
  };

  /**
   * @return the input realizations
   * @throw when no samples are available
   */
  const gauss::TrainSet &GetSamplesInput() const;
  /**
   * @return the output realizations
   * @throw when no samples are available
   */
  const gauss::TrainSet &GetSamplesOutput() const;

private:
  std::unique_ptr<gauss::TrainSet> input;
  std::unique_ptr<gauss::TrainSet> output;
};
} // namespace gauss::gp
