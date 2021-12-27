/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/GaussianProcessVectorial.h>
#include <fstream>
#include <functional>

std::vector<Eigen::VectorXd> get_equispaced_samples(const double interval_min,
                                                    const double interval_max,
                                                    const std::size_t points) {
  const double delta =
      (interval_max - interval_min) / static_cast<double>(points - 1);
  std::vector<Eigen::VectorXd> result;
  result.reserve(points);
  double point = interval_min;
  {
    Eigen::VectorXd temp(1);
    temp << point;
    result.emplace_back(temp);
  }
  point += delta;
  for (std::size_t p = 1; p < points; ++p, point += delta) {
    Eigen::VectorXd temp(1);
    temp << point;
    result.emplace_back(temp);
  }
  return result;
}

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

using LogPrediction = gauss::gp::GaussianProcessVectorial::Prediction;

void log_predictions(
    const std::vector<Eigen::VectorXd> &input_predictions,
    const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> real_function,
    const std::vector<LogPrediction> &predictions,
    const std::string &file_name) {
  std::ofstream log(file_name);
  auto it_in = input_predictions.begin();
  for (auto it_out = predictions.begin(); it_out != predictions.end();
       ++it_out, ++it_in) {
    log << it_in->transpose() << ' ' << real_function(*it_in).transpose() << ' '
        << it_out->mean.transpose() << ' '
        << it_out->mean(0) - 2.5 * sqrt(it_out->covariance) << ' '
        << it_out->mean(0) + 2.5 * sqrt(it_out->covariance) << std::endl;
  }
}
