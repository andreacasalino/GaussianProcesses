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
#include "LogUtils.h"

#include <functional>
#include <iostream>
#include <unordered_map>

double polinomial_fun(const Eigen::VectorXd &point);

double periodic_fun(const Eigen::VectorXd &point);

int main() {
  using Function = std::function<double(const Eigen::Vector2d &)>;

  // set of functions to approximate using gaussian processes
  std::unordered_map<std::string, Function> functions_to_approximate;
  functions_to_approximate.emplace(
      "polinomial_function",
      [](const Eigen::VectorXd &point) { return polinomial_fun(point); });
  functions_to_approximate.emplace(
      "periodic_function",
      [](const Eigen::VectorXd &point) { return periodic_fun(point); });
  functions_to_approximate.emplace(
      "composite_function", [](const Eigen::VectorXd &point) {
        return polinomial_fun(point) + periodic_fun(point);
      });

  // equispaced grid of points forming the training set for the
  // gaussian process
  const auto input_samples =
      gauss::gp::samples::grid({-6.0, -6.0}, {6.0, 6.0}, 15);

  // equispaced grid of points where the prediction of the gaussian process will
  // be checked to be similar to the function where the training set samples
  // were taken
  const auto input_for_predictions =
      gauss::gp::samples::grid({-6.0, -6.0}, {6.0, 6.0}, 75);

  nlohmann::json log_json;

  for (auto &[title, function] : functions_to_approximate) {
    // an empty gaussian process is initially built
    gauss::gp::GaussianProcessVectorial<2, 1> gauss_proc(
        std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0));
    gauss_proc.setWhiteNoiseStandardDeviation(0.001);

    // fill the training set of the gaussian process
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

    // predict values using the gaussian process
    std::vector<std::vector<double>> prediction_uncertainties;
    std::vector<std::vector<double>> prediction_means;
    std::vector<std::vector<double>> expected_means;
    for (const auto &row : input_for_predictions) {
      auto &prediction_uncertainties_row =
          prediction_uncertainties.emplace_back();
      auto &prediction_means_row = prediction_means.emplace_back();
      auto &expected_means_row = expected_means.emplace_back();
      for (const auto point : row) {
        auto prediction =
            gauss_proc.predict(point, gauss::gp::RAW_VALUES_PREDICTION_TAG);
        prediction_uncertainties_row.push_back(sqrt(prediction.covariance));
        prediction_means_row.push_back(prediction.mean(0));
        expected_means_row.push_back(function(point));
      }
    }

    // log the results
    auto &new_log = log_json[title];
    auto &train_set = new_log["train_set"];
    gauss::gp::samples::load(train_set["inputs"], input_samples);
    train_set["outputs"] = output_samples;
    auto &pred = new_log["predictions"];
    gauss::gp::samples::load(pred["inputs"], input_for_predictions);
    pred["means"] = prediction_means;
    pred["expected"] = expected_means;
    pred["sigmas"] = prediction_uncertainties;

    std::cout << "call 'python Visualize-2D.py Log.json " << title
              << "' to visualize the results" << std::endl;
  }

  gauss::gp::samples::print(log_json, "Log.json");

  return EXIT_SUCCESS;
}

double polinomial_fun(const Eigen::VectorXd &point) {
  return point(0) * 0.5 - point(1) * point(1) * 0.1 +
         point(0) * point(0) * 0.05;
}

double periodic_fun(const Eigen::VectorXd &point) { return sin(point.norm()); }
