/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/Error.h>
#include <GaussianProcess/TrainSet.h>

namespace gauss::gp {
namespace {
Eigen::VectorXd get_slice(const Eigen::VectorXd &source,
                          const Eigen::Index &start, const Eigen::Index &end) {
  Eigen::VectorXd result(end - start);
  Eigen::Index i = 0;
  for (auto p = start; p < end; ++p, ++i) {
    result(i) = source(p);
  }
  return result;
}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>>
import_train_set(const std::string &file_to_read,
                 const std::size_t input_space_size) {
  gauss::TrainSet temp(file_to_read);
  if (static_cast<std::size_t>(temp.GetSamples().front().size()) <
      input_space_size) {
    throw gauss::gp::Error("Invalid size of samples in parsed files");
  }
  std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::VectorXd>> result;
  auto &input = result.first;
  auto &output = result.second;
  for (const auto &vec : temp.GetSamples()) {
    input.push_back(get_slice(vec, 0, input_space_size));
    output.push_back(get_slice(vec, input_space_size + 1, vec.size()));
  }
  return result;
}
} // namespace

TrainSet::TrainSet(const std::size_t input_space_size,
                   const std::size_t output_space_size)
    : input_space_size(input_space_size), output_space_size(output_space_size) {
}

TrainSet::TrainSet(const gauss::TrainSet &inputSamples,
                   const gauss::TrainSet &outputSamples) {
  this->input_space_size = inputSamples.GetSamples().front().size();
  this->output_space_size = outputSamples.GetSamples().front().size();
  input_samples = inputSamples.GetSamples();
  output_samples = outputSamples.GetSamples();
}

TrainSet::TrainSet(const std::string &file_to_read,
                   const std::size_t input_space_size) {
  auto samples = import_train_set(file_to_read, input_space_size);
  this->input_space_size = samples.first.front().size();
  this->output_space_size = samples.second.front().size();
  input_samples = std::move(samples.first);
  output_samples = std::move(samples.second);
}

void TrainSet::addSample(const Eigen::VectorXd &input_sample,
                         const Eigen::VectorXd &output_sample) {
  if (input_sample.size() != input_space_size) {
    throw Error{"Invalid new sample"};
  }
  input_samples.push_back(input_sample);

  if (output_sample.size() != output_space_size) {
    throw Error{"Invalid new sample"};
  }
  output_samples.push_back(output_sample);
}

void TrainSet::addSample(const Eigen::VectorXd &sample) {
  if ((input_space_size + output_space_size) !=
      (getInputStateSpaceSize() + getOutputStateSpaceSize())) {
    throw gauss::gp::Error("Invalid new sample");
  }
  input_samples.emplace_back(get_slice(sample, 0, getInputStateSpaceSize()));
  output_samples.emplace_back(
      get_slice(sample, getInputStateSpaceSize() + 1, sample.size()));
}

void TrainSet::addSamples(const gauss::gp::TrainSet &o) {
  if ((input_space_size != o.input_space_size) ||
      (output_space_size != o.output_space_size)) {
    throw Error{"Invalid new samples"};
  }
  for (std::size_t k = 0; k < o.input_samples.size(); ++k) {
    input_samples.push_back(o.input_samples[k]);
    output_samples.push_back(o.output_samples[k]);
  }
}
} // namespace gauss::gp
