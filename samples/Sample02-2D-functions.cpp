/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

// just a bunch of functionalities to generate and visualize the predictions
// made by gaussian processes
#include "Utils.h"

#include <iostream>
#include <unordered_map>

void convert(nlohmann::json &recipient,
             const std::vector<std::vector<Eigen::VectorXd>> &grid_2D);

int main() {
  std::unordered_map<std::string, gauss::gp::samples::Function>
      functions_to_approximate;
  functions_to_approximate.emplace(
      "polinomial_function", [](const Eigen::VectorXd &point) {
        const auto val = point.norm() + 12.0;
        return 0.05 * pow(val, 3.0) - 0.1 * pow(val, 2.0);
      });
  functions_to_approximate.emplace(
      "periodic_function",
      [](const Eigen::VectorXd &point) { return 3.0 * sin(point.norm()); });
  functions_to_approximate.emplace(
      "composite_function", [](const Eigen::VectorXd &point) {
        const auto val = point.norm();
        return 0.05 * pow(val + 12, 3.0) - 0.1 * pow(val + 12, 2.0) +
               3.0 * sin(val);
      });

  const auto input_samples = gauss::gp::samples::make_equispaced_input_samples(
      {-6.0, -6.0}, {6.0, 6.0}, 20);

  const auto input_for_predictions =
      gauss::gp::samples::make_equispaced_input_samples({-6.0, -6.0},
                                                        {6.0, 6.0}, 75);

  nlohmann::json log_json;

  for (auto &[title, function] : functions_to_approximate) {
    gauss::gp::GaussianProcessVectorial<2, 1> gauss_proc(
        std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0));
    gauss_proc.setWhiteNoiseStandardDeviation(0.001);

    std::vector<std::vector<double>> output_samples;
    for (const auto &row : input_samples) {
      auto &output_samples_row = output_samples.emplace_back();
      for (const auto input_sample : row) {
        output_samples_row.push_back(function(input_sample));
        Eigen::VectorXd output_sample(1);
        output_sample << output_samples_row.back();
        gauss_proc.getTrainSet().addSample(input_sample, output_sample);
      }
    }

    std::vector<std::vector<double>> prediction_uncertainties;
    std::vector<std::vector<double>> prediction_means;
    std::vector<std::vector<double>> expected_means;

    for (const auto &row : input_for_predictions) {
      auto &prediction_uncertainties_row =
          prediction_uncertainties.emplace_back();
      auto &prediction_means_row = prediction_means.emplace_back();
      auto &expected_means_row = expected_means.emplace_back();
      for (const auto point : row) {
        auto prediction = gauss_proc.predict2(point);
        prediction_uncertainties_row.push_back(sqrt(prediction.covariance));
        prediction_means_row.push_back(prediction.mean(0));
        expected_means_row.push_back(function(point));
      }
    }

    // log the results
    auto &new_log = log_json[title];
    auto &train_set = new_log["train_set"];
    convert(train_set["inputs"], input_samples);
    train_set["outputs"] = output_samples;
    auto &pred = new_log["predictions"];
    convert(pred["inputs"], input_for_predictions);
    pred["means"] = prediction_means;
    pred["expected"] = expected_means;
    pred["sigmas"] = prediction_uncertainties;

    std::cout << "call 'python Visualize-2D.py Log.json " << title
              << "' to visualize the results" << std::endl;
  }

  gauss::gp::samples::print(log_json, "Log.json");

  return EXIT_SUCCESS;
}

void convert(nlohmann::json &recipient,
             const std::vector<std::vector<Eigen::VectorXd>> &grid_2D) {
  std::vector<std::vector<double>> xs, ys;
  for (const auto &row : grid_2D) {
    auto &xs_row = xs.emplace_back();
    auto &ys_row = ys.emplace_back();
    for (const auto &point : row) {
      xs_row.push_back(point(0));
      ys_row.push_back(point(1));
    }
  }
  recipient["x"] = xs;
  recipient["y"] = ys;
}
