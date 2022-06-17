/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/SpaceSizesAware.h>
#include <GaussianUtils/TrainSet.h>

namespace gauss::gp {
class TrainSet : public SpaceSizesAware {
public:
  virtual ~TrainSet() = default;

  TrainSet(const TrainSet &o) = default;
  TrainSet &operator=(const TrainSet &o) = default;

  TrainSet(TrainSet &&o) = default;
  TrainSet &operator=(TrainSet &&o) = default;

  /**
   * @brief An empty train set with no samples is actually built.
   */
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
   * @brief adds a new pair of input-output samples.
   * @throw in case the sizes of the input or the output sample are inconsistent
   * with the input output sizes of the training set.
   */
  void addSample(const Eigen::VectorXd &input_sample,
                 const Eigen::VectorXd &output_sample);
  /**
   * @throw in case the sizes of the training set to absorb are inconsistent
   * with the input output sizes of the training set.
   */
  void addSamples(const gauss::gp::TrainSet &o);

  /**
   * @brief similar to TrainSet::addSample, but passing a single vector whose
   * first components are the ones pertaining to the input and the others to the
   * output.
   * Components are distingusihed according to the sizes of the trainig set
   * (defined when building the train set object).
   * @throw in case the size of the passed vector is inconsistent
   * with the input output sizes of the training set.
   */
  void addSample(const Eigen::VectorXd &sample);

  /**
   * @return the input realizations
   */
  const std::vector<Eigen::VectorXd> &GetSamplesInput() const {
    return input_samples;
  }
  /**
   * @return the output realizations
   */
  const std::vector<Eigen::VectorXd> &GetSamplesOutput() const {
    return output_samples;
  }

private:
  std::vector<Eigen::VectorXd> input_samples;
  std::vector<Eigen::VectorXd> output_samples;
};

/**
 * @brief Creates a new Train Set object, with
 * input_samples and output_samples red from a file.
 * In each row of such a file, the initial columns represent the input
 * realization, while the remaining the output one. The size of the input
 * samples are assumed equal to input_space_size
 *
 * @param file_to_read
 * @param input_space_size
 * @throw In case the file is not existent
 * @throw In case the file contains less columns than input_space_size
 */
TrainSet import_train_set(const std::string &file_to_read,
                          const std::size_t input_space_size);

class TrainSetAware {
public:
  virtual ~TrainSetAware() = default;

  virtual const TrainSet &getTrainSet() const = 0;

protected:
  TrainSetAware() = default;
};
} // namespace gauss::gp
