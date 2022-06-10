/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "KernelVisualizer.h"

using namespace nlohmann;

namespace gauss::gp {
std::vector<double>
convert_samples(const std::vector<Eigen::VectorXd> &samples) {
  std::vector<double> result;
  result.reserve(samples.size());
  for (const auto &sample : samples) {
    result.push_back(sample(0));
  }
  return result;
}

namespace {
std::vector<Eigen::VectorXd> make_equi_samples(const std::size_t samples_numb) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples_numb);
  double val = -1.0;
  const double delta = 2.0 / static_cast<double>(samples_numb - 1);
  for (std::size_t k = 0; k < samples_numb; ++k) {
    auto &new_sample = result.emplace_back(1);
    new_sample << val;
    val += delta;
  }
  return result;
}
} // namespace

nlohmann::json make_kernel_viz_log(const std::size_t samples_numb,
                                   const KernelFunction &kernel,
                                   const std::string &title,
                                   const std::string &tag) {
  const auto samples = make_equi_samples(samples_numb);

  std::vector<std::vector<double>> kernel_matrix;
  kernel_matrix.reserve(samples_numb);
  for (Eigen::Index r = 0; r < samples_numb; ++r) {
    auto &row = kernel_matrix.emplace_back();
    row.reserve(samples_numb);
    for (Eigen::Index c = 0; c < samples_numb; ++c) {
      row.push_back(kernel.evaluate(samples[r], samples[c]));
    }
  }

  nlohmann::json recipient;
  recipient["samples"] = convert_samples(samples);
  recipient["kernel"] = kernel_matrix;
  recipient["title"] = title;
  recipient["tag"] = tag;
  return recipient;
}
} // namespace gauss::gp
