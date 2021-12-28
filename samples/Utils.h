/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/GaussianProcess.h>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>

std::vector<Eigen::VectorXd> get_output_samples(
    const std::vector<Eigen::VectorXd> &input_samples,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &function) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(input_samples.size());
  for (const auto &sample : input_samples) {
    result.emplace_back(function(sample));
  }
  return result;
}

std::vector<double> get_equispaced_samples(const double interval_min,
                                           const double interval_max,
                                           const std::size_t points) {
  const double delta =
      (interval_max - interval_min) / static_cast<double>(points - 1);
  std::vector<double> result;
  result.reserve(points);
  double point = interval_min;
  result.push_back(point);
  point += delta;
  for (std::size_t p = 1; p < points; ++p, point += delta) {
    result.push_back(point);
  }
  return result;
}

Eigen::VectorXd convert(const double &value) {
  Eigen::VectorXd result(1);
  result << value;
  return result;
}

std::vector<Eigen::VectorXd> convert(const std::vector<double> &values) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(values.size());
  for (const auto &value : values) {
    result.push_back(convert(value));
  }
  return result;
}

std::string to_json(const std::vector<double> &data) {
  std::stringstream stream;
  stream << '[';
  auto it = data.begin();
  stream << *it;
  ++it;
  for (it; it != data.end(); ++it) {
    stream << ',' << *it;
  }
  stream << ']';
  return stream.str();
}

std::string to_json(const std::vector<std::vector<double>> &data) {
  std::stringstream stream;
  stream << '[';
  auto it = data.begin();
  stream << to_json(*it);
  ++it;
  for (it; it != data.end(); ++it) {
    stream << ',' << to_json(*it);
  }
  stream << ']';
  return stream.str();
}

std::string to_json(const std::map<std::string, std::string> &fields) {
  std::stringstream stream;
  stream << '{';
  std::size_t pos = 0;
  for (const auto &[name, val] : fields) {
    if (pos > 0) {
      stream << ',';
    }
    stream << '\"' << name << '\"' << ':' << val;
    ++pos;
  }
  stream << '}';
  return stream.str();
}

std::vector<Eigen::VectorXd> get_input_samples(const double ray,
                                               const std::size_t samples) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples);
  for (std::size_t s = 0; s < samples; ++s) {
    result.emplace_back(2);
    result.back().setRandom();
    result.back() *= ray;
  }
  return result;
};

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_grid(const double ray, const std::size_t samples_along_axis) {
  std::vector<std::vector<double>> x_coord, y_coord;
  x_coord.reserve(samples_along_axis);
  y_coord.reserve(samples_along_axis);
  auto x_samples = get_equispaced_samples(-ray, ray, samples_along_axis);
  auto y_samples = get_equispaced_samples(-ray, ray, samples_along_axis);
  for (std::size_t r = 0; r < samples_along_axis; ++r) {
    x_coord.emplace_back();
    x_coord.back().reserve(samples_along_axis);
    y_coord.emplace_back();
    y_coord.back().reserve(samples_along_axis);
    for (std::size_t c = 0; c < samples_along_axis; ++c) {
      x_coord.back().push_back(x_samples[r]);
      y_coord.back().push_back(y_samples[c]);
    }
  }
  return std::make_pair<std::vector<std::vector<double>>,
                        std::vector<std::vector<double>>>(std::move(x_coord),
                                                          std::move(y_coord));
}

class Logger {
public:
  void add_field(const std::vector<double> &data, const std::string &name) {
    fields.emplace(name, to_json(data));
  }

  void print(const std::string &file_name) const {
    std::ofstream stream(file_name);
    stream << to_json(fields);
  }

protected:
  std::map<std::string, std::string> fields;
};

class LoggerExtra : public Logger {
public:
  void add_field(const std::vector<std::vector<double>> &data,
                 const std::string &name) {
    fields.emplace(name, to_json(data));
  }

  void add_samples(const std::vector<Eigen::VectorXd> &input_samples,
                   const std::vector<double> &output_samples) {
    fields.emplace("output_samples", to_json(output_samples));
    std::vector<double> temp;
    temp.reserve(input_samples.size());
    for (const auto &sample : input_samples) {
      temp.push_back(sample(0));
    }
    fields.emplace("input_x_samples", to_json(temp));
    temp.clear();
    for (const auto &sample : input_samples) {
      temp.push_back(sample(1));
    }
    fields.emplace("input_y_samples", to_json(temp));
  }
};
