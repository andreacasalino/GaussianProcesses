/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "LogUtils.h"

#include <fstream>

namespace gauss::gp::samples {
void load(nlohmann::json &recipient,
          const std::vector<Eigen::VectorXd> &subject) {
  std::vector<double> as_vec;
  for (const auto &sample : subject) {
    as_vec.push_back(sample(0));
  }
  recipient = as_vec;
}

void load(nlohmann::json &recipient,
          const std::vector<std::vector<Eigen::Vector2d>> &subject) {
  std::vector<std::vector<double>> x_coordinates;
  std::vector<std::vector<double>> y_coordinates;
  for (const auto row : subject) {
    auto x_row = x_coordinates.emplace_back();
    auto y_row = y_coordinates.emplace_back();
    for (const auto point : row) {
      x_row.push_back(point.x());
      y_row.push_back(point.y());
    }
  }
  recipient["x"] = x_coordinates;
  recipient["y"] = y_coordinates;
}

void print(const nlohmann::json &subject, const std::string &file_name) {
  std::ofstream stream(file_name);
  stream << subject.dump();
}
} // namespace gauss::gp::samples
