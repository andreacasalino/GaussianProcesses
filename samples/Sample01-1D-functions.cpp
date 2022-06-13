/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/PeriodicFunction.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

// just a bunch of functionalities to generate and visualize the predictions
// made by gaussian processes
#include "Utils.h"

#include <iostream>
#include <unordered_map>

int main() {
  std::unordered_map<std::string, std::pair<gauss::gp::samples::Function,
                                            gauss::gp::KernelFunctionPtr>>
      functions_to_approximate;

  functions_to_approximate.emplace(
      "polinomial_function",
      std::make_pair(
          [](const Eigen::VectorXd &point) {
            return 0; // TODO
          },
          std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0)));

  functions_to_approximate.emplace(
      "periodic_function",
      std::make_pair(
          [](const Eigen::VectorXd &point) { return 3.0 * sin(point(0)); },
          std::make_unique<gauss::gp::PeriodicFunction>(1.0, 1.0, 3.0)));

  const auto input_samples =
      gauss::gp::samples::make_equispaced_input_samples(-6.0, 6.0, 15);
  const auto output_samples = gauss::gp::samples::make_output_samples(
      [](const Eigen::VectorXd &input) {
        return 0; // TODO
      },
      input_samples);

  const auto input_for_predictions =
      gauss::gp::samples::make_equispaced_input_samples(-6.0, 6.0, 100);

  nlohmann::json log_json;

  for (auto &[title, data] : functions_to_approximate) {
    auto &[function, kernel] = data;
    gauss::gp::GaussianProcess gauss_proc(std::move(kernel), 1, 1);

    std::vector<double> prediction_uncertainties;
    std::vector<Eigen::VectorXd> prediction_means;

    for (const auto &point : input_for_predictions) {
      auto prediction = gauss_proc.predict2(point);
      prediction_uncertainties.push_back(sqrt(prediction.covariance));
      prediction_means.push_back(prediction.mean);
    }

    // log the results
    auto &new_log = log_json[title];
    auto &train_set = new_log["train_set"];
    gauss::gp::samples::convert(train_set["inputs"], input_samples);
    gauss::gp::samples::convert(train_set["outputs"], output_samples);
    auto &pred = new_log["predictions"];
    gauss::gp::samples::convert(pred["inputs"], input_for_predictions);
    gauss::gp::samples::convert(pred["means"], prediction_means);
    pred["sigmas"] = prediction_uncertainties;

    std::cout << "call 'python Visualize.py Log.json " << title
              << "' to visualize the results" << std::endl;
  }

  gauss::gp::samples::print(log_json, "Log.json");

  return EXIT_SUCCESS;
}
