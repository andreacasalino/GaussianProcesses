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

int main() {
  std::unordered_map<std::string, gauss::gp::samples::Function>
      functions_to_approximate;
  functions_to_approximate.emplace(
      "polinomial_function", [](const Eigen::VectorXd &point) {
        return 0.05 * pow(point(0), 3.0) - 0.1 * pow(point(0), 2.0);
      });
  functions_to_approximate.emplace(
      "periodic_function",
      [](const Eigen::VectorXd &point) { return 3.0 * sin(point(0)); });
  functions_to_approximate.emplace(
      "composite_function", [](const Eigen::VectorXd &point) {
        return 0.05 * pow(point(0), 3.0) - 0.1 * pow(point(0), 2.0) +
               3.0 * sin(point(0));
      });

  const auto input_samples =
      gauss::gp::samples::make_equispaced_input_samples(-6.0, 6.0, 15);

  const auto input_for_predictions =
      gauss::gp::samples::make_equispaced_input_samples(-6.0, 6.0, 250);

  nlohmann::json log_json;

  for (auto &[title, function] : functions_to_approximate) {
    gauss::gp::GaussianProcess gauss_proc(
        std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0), 1, 1);
    gauss_proc.setWhiteNoiseStandardDeviation(0.001);

    const auto output_samples =
        gauss::gp::samples::make_output_samples(function, input_samples);
    for (std::size_t k = 0; k < input_samples.size(); ++k) {
      gauss_proc.getTrainSet().addSample(input_samples[k], output_samples[k]);
    }

    std::vector<double> prediction_uncertainties;
    std::vector<double> prediction_means;

    for (const auto &point : input_for_predictions) {
      auto prediction = gauss_proc.predict2(point);
      prediction_uncertainties.push_back(sqrt(prediction.covariance));
      prediction_means.push_back(prediction.mean(0));
    }

    // log the results
    auto &new_log = log_json[title];
    auto &train_set = new_log["train_set"];
    gauss::gp::samples::convert(train_set["inputs"], input_samples);
    gauss::gp::samples::convert(train_set["outputs"], output_samples);
    auto &pred = new_log["predictions"];
    gauss::gp::samples::convert(pred["inputs"], input_for_predictions);
    pred["means"] = prediction_means;
    pred["sigmas"] = prediction_uncertainties;

    std::cout << "call 'python Visualize.py Log.json " << title
              << "' to visualize the results" << std::endl;
  }

  gauss::gp::samples::print(log_json, "Log.json");

  return EXIT_SUCCESS;
}
