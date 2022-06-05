/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianUtils/TrainSet.h>

namespace gauss::gp {
class TrainSet {
public:
  virtual ~TrainSet() = default;

  TrainSet(const TrainSet &o) = default;
  TrainSet &operator=(const TrainSet &o) = default;

  TrainSet(TrainSet &&o) = default;
  TrainSet &operator=(TrainSet &&o) = default;

  TrainSet(const std::size_t input_space_size,
           const std::size_t output_space_size);

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
  TrainSet(const gauss::TrainSet &input_samples,
           const gauss::TrainSet &output_samples);

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

  void addSample(const Eigen::VectorXd &input_sample,
                 const Eigen::VectorXd &output_sample);

  void addSamples(const gauss::gp::TrainSet &o);

  /**
   * @brief Add a new pair input-output sample.
   * The values of the input part is assumed as the initial values of the passed
   * vector, while the remaining values are assumed to make the output part.
   *
   * @param sample
   */
  void addSample(const Eigen::VectorXd &sample);

  std::size_t getInputStateSpaceSize() const {
    return static_cast<std::size_t>(input_space_size);
  };
  std::size_t getOutputStateSpaceSize() const {
    return static_cast<std::size_t>(output_space_size);
  };

  /**
   * @return the input realizations
   * @throw when no samples are available
   */
  const std::vector<Eigen::VectorXd> &GetSamplesInput() const {
    return input_samples;
  }
  /**
   * @return the output realizations
   * @throw when no samples are available
   */
  const std::vector<Eigen::VectorXd> &GetSamplesOutput() const {
    return output_samples;
  }

private:
  Eigen::Index input_space_size;
  std::vector<Eigen::VectorXd> input_samples;

  Eigen::Index output_space_size;
  std::vector<Eigen::VectorXd> output_samples;
};

class TrainSetAware {
public:
  virtual const TrainSet &getTrainSet() const = 0;

protected:
  TrainSetAware() = default;
};
} // namespace gauss::gp
