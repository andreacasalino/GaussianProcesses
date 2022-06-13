/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Utils.h"

#include <fstream>

namespace gauss::gp::samples {
namespace {
std::vector<double> linspace(const double min, const double max,
                             const std::size_t size) {
  const double delta = (max - min) / static_cast<double>(size - 1);
  double val = min;
  std::vector<double> result;
  result.reserve(size);
  for (std::size_t k = 0; k < size; ++k, val += delta) {
    result.push_back(val);
  }
  return result;
}

void fill(std::vector<Eigen::VectorXd> &recipient,
          const std::vector<std::vector<double>> &intervals, std::size_t index,
          const std::vector<double> &cumulated) {
  if ((index + 1) == intervals.size()) {
    for (const auto &val : intervals[index]) {
      auto cumulated_next = cumulated;
      cumulated_next.push_back(val);
      Eigen::VectorXd point(cumulated_next.size());
      point.setZero();
      for (std::size_t k = 0; k < cumulated_next.size(); ++k) {
        point(k) = cumulated_next[k];
      }
      recipient.push_back(point);
    }
    return;
  }
  for (const auto &val : intervals[index]) {
    auto cumulated_next = cumulated;
    cumulated_next.push_back(val);
    fill(recipient, intervals, index + 1, cumulated_next);
  }
}
} // namespace

std::vector<Eigen::VectorXd>
make_equispaced_input_samples(const double min, const double max,
                              const std::size_t size) {
  std::vector<Eigen::VectorXd> result;
  fill(result, {linspace(min, max, size)}, 0, std::vector<double>{});
  return result;
}

std::vector<Eigen::VectorXd>
make_equispaced_input_samples(const Eigen::VectorXd &min,
                              const Eigen::VectorXd &max,
                              const std::size_t size) {
  std::vector<std::vector<double>> intervals;

  std::vector<Eigen::VectorXd> result;
  fill(result, intervals, 0, std::vector<double>{});
  return result;
}

std::vector<Eigen::VectorXd>
make_output_samples(const Function &function,
                    const std::vector<Eigen::VectorXd> &input_samples) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(input_samples.size());
  for (const auto &sample : input_samples) {
    Eigen::VectorXd temp(1);
    temp << function(sample);
    result.push_back(temp);
  }
  return result;
}

void convert(nlohmann::json &recipient,
             const std::vector<Eigen::VectorXd> &subject) {
  if (subject.front().size() == 1) {
    std::vector<double> as_vec;
    for (const auto &sample : subject) {
      as_vec.push_back(sample(0));
    }
    recipient = as_vec;
  } else {
    recipient = nlohmann::json::array();
    for (const auto v : subject) {
      std::vector<double> as_vec;
      for (Eigen::Index k = 0; k < v.size(); ++k) {
        as_vec.push_back(v(k));
      }
      recipient.push_back(as_vec);
    }
  }
}

void print(const nlohmann::json &subject, const std::string &file_name) {
  std::ofstream stream(file_name);
  stream << subject.dump();
}
} // namespace gauss::gp::samples
