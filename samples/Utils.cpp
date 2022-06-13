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
} // namespace

std::vector<Eigen::VectorXd>
make_equispaced_input_samples(const double min, const double max,
                              const std::size_t size) {
  std::vector<Eigen::VectorXd> result;
  for (const auto &val : linspace(min, max, size)) {
    result.emplace_back(1) << val;
  }
  return result;
}

std::vector<std::vector<Eigen::VectorXd>>
make_equispaced_input_samples(const std::array<double, 2> &min,
                              const std::array<double, 2> &max,
                              const std::size_t size) {
  const auto xs = linspace(min[0], max[0], size);
  const auto ys = linspace(min[1], max[1], size);
  std::vector<std::vector<Eigen::VectorXd>> result;
  for (const auto &x : xs) {
    auto &row = result.emplace_back();
    for (const auto &y : ys) {
      row.emplace_back(2) << x, y;
    }
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
