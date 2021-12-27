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
#include <sstream>

std::vector<Eigen::VectorXd> get_input_samples(const double ray,
                                               const std::size_t points) {
  std::vector<Eigen::VectorXd> result;
  for (std::size_t s = 0; s < points; ++s) {
    result.emplace_back(2);
    result.back().setRandom();
    result.back() *= ray;
  }
  return result;
};

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

std::vector<std::vector<Eigen::VectorXd>>
get_equispaced_grid_samples(const double ray, const std::size_t points) {
  std::vector<std::vector<Eigen::VectorXd>> result;
  result.reserve(points);
  auto x_vals = get_equispaced_samples(-ray, ray, points);
  auto y_vals = get_equispaced_samples(-ray, ray, points);
  for (const auto &x : x_vals) {
    result.emplace_back();
    result.back().reserve(points);
    for (const auto &y : y_vals) {
      Eigen::VectorXd point(2);
      point << x, y;
      result.back().push_back(point);
    }
  }
  return result;
}

using LogPrediction = gauss::gp::GaussianProcess::Prediction;
std::vector<std::vector<LogPrediction>>
get_predictions(const std::vector<std::vector<Eigen::VectorXd>> &input_grid,
                gauss::gp::GaussianProcess &process) {
  std::vector<std::vector<LogPrediction>> result;
  result.reserve(input_grid.size());
  for (const auto &samples : input_grid) {
    result.emplace_back();
    result.back().reserve(samples.size());
    for (const auto &sample : samples) {
      result.back().push_back(process.predict2(sample));
    }
  }
  return result;
}

class Logger {
public:
  void print(const std::string &file_name) const {
    std::stringstream stream;
    stream << '{';
    stream << "\"samples\":" << samples_json->str();
    stream << ",\"predictions\":" << predictions_json->str();
    stream << '}';
    std::ofstream file;
    file << stream.str();
  };

  void log_samples(const std::vector<Eigen::VectorXd> &input_samples,
                   const std::vector<Eigen::VectorXd> &output_samples) {
    samples_json = std::make_unique<std::stringstream>();
    *samples_json << '{';
    *samples_json << "\"inputs\":";
    log_vector<Eigen::VectorXd>(*samples_json, input_samples);
    *samples_json << "\"outputs\":";
    log_vector<Eigen::VectorXd>(*samples_json, output_samples);
    *samples_json << '}';
  };

  void
  log_predictions(const std::vector<std::vector<Eigen::VectorXd>> &points,
                  const std::vector<std::vector<LogPrediction>> &predictions) {
    predictions_json = std::make_unique<std::stringstream>();
    *predictions_json << '{';
    *predictions_json << "\"x_coord\":";
    log_matrix<Eigen::VectorXd>(*predictions_json, points, 0);
    *predictions_json << ",\"y_coord\":";
    log_matrix<Eigen::VectorXd>(*predictions_json, points, 1);
    *predictions_json << ",\"predictions\":";
    log_matrix<LogPrediction>(*predictions_json, predictions, true);
    *predictions_json << ",\"covariances\":";
    log_matrix<LogPrediction>(*predictions_json, predictions, false);
    *predictions_json << '}';
  }

private:
  static void log_(std::stringstream &stream, const double &v) { stream << v; }

  static void log_(std::stringstream &stream, const Eigen::VectorXd &v) {
    stream << '[';
    stream << v(0);
    for (Eigen::Index i = 1; i < v.size(); ++i) {
      stream << ',' << v(i);
    }
    stream << ']';
  }

  static void log_(std::stringstream &stream, const Eigen::VectorXd &v,
                   const Eigen::Index i) {
    stream << v(i);
  }

  static void log_(std::stringstream &stream, const LogPrediction &p,
                   bool mean_or_cov) {
    if (mean_or_cov) {
      stream << p.mean;
    } else {
      stream << p.covariance;
    }
  }

  template <typename T, typename... Args>
  static void log_vector(std::stringstream &stream,
                         const std::vector<T> &values, Args... args) {
    stream << '[';
    auto it = values.begin();
    log_(stream, *it, args...);
    ++it;
    for (it; it != values.end(); ++it) {
      stream << ',';
      log_(stream, *it, args...);
    }
    stream << ']';
  }

  template <typename T, typename... Args>
  static void log_matrix(std::stringstream &stream,
                         const std::vector<std::vector<T>> &values,
                         Args... args) {
    stream << '[';
    auto it = values.begin();
    log_vector(stream, *it, args...);
    ++it;
    for (it; it != values.end(); ++it) {
      stream << ',';
      log_vector(stream, *it, args...);
    }
    stream << ']';
  }

  std::unique_ptr<std::stringstream> samples_json;
  std::unique_ptr<std::stringstream> predictions_json;
};
