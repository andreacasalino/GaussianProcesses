/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include <TrainingTools/iterative/solvers/GradientDescend.h>

// just a bunch of functionalities to generate and visualize the predictions
// made by gaussian processes
#include "LogUtils.h"

#include <iostream>
#include <unordered_map>

double function_to_approximate(const Eigen::VectorXd &point);

void log_predictions(nlohmann::json &recipient,
                     const gauss::gp::GaussianProcess &process,
                     const std::vector<Eigen::VectorXd> &input_for_predictions);

int main() {
  gauss::gp::GaussianProcessScalar<1> gauss_proc(
      std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0));
  gauss_proc.setWhiteNoiseStandardDeviation(0.001);
  // fill the gaussian process with samples
  // in the interval (6,6)
  for (const auto sample_in : gauss::gp::samples::linspace(-6.0, 6.0, 15)) {
    Eigen::VectorXd sample_out(1);
    sample_out << function_to_approximate(sample_in);
    gauss_proc.getTrainSet().addSample(sample_in, sample_out);
  }

  // equispaced grid of points where the prediction of the gaussian process will
  // be checked
  const auto input_for_predictions =
      gauss::gp::samples::linspace(-6.0, 6.0, 250);

  nlohmann::json log_json;

  // check predictive performance of un-trained model
  std::cout << "un trained model logarithic likelihood: "
            << gauss_proc.getLogLikelihood() << std::endl;
  log_predictions(log_json["un-trained"], gauss_proc, input_for_predictions);

  // tune the hyperparameters of the model
  ::train::GradientDescendFixed trainer;
  trainer.setOptimizationStep(0.05);
  trainer.setMaxIterations(10);
  gauss::gp::train(gauss_proc, trainer);

  // check predictive performance of trained model
  std::cout << "trained model logarithic likelihood: "
            << gauss_proc.getLogLikelihood() << std::endl;
  log_predictions(log_json["trained"], gauss_proc, input_for_predictions);

  gauss::gp::samples::print(log_json, "Log.json");

  std::cout << "call 'python Visualize-train.py "
               "Log.json' to visualize the results"
            << std::endl;

  return EXIT_SUCCESS;
}

double function_to_approximate(const Eigen::VectorXd &point) {
  return 0.05 * pow(point(0), 3.0) - 0.1 * pow(point(0), 2.0) +
         3.0 * sin(point(0));
}

void log_predictions(
    nlohmann::json &recipient, const gauss::gp::GaussianProcess &process,
    const std::vector<Eigen::VectorXd> &input_for_predictions) {
  std::vector<double> prediction_uncertainties;
  std::vector<double> prediction_means;
  std::vector<double> expected_means;
  for (const auto &point : input_for_predictions) {
    auto prediction = process.predict2(point);
    prediction_uncertainties.push_back(sqrt(prediction.covariance));
    prediction_means.push_back(prediction.mean(0));
    expected_means.push_back(function_to_approximate(point));
  }
  // log the results
  gauss::gp::samples::load(recipient["inputs"], input_for_predictions);
  recipient["expected"] = expected_means;
  recipient["means"] = prediction_means;
  recipient["sigmas"] = prediction_uncertainties;
}
