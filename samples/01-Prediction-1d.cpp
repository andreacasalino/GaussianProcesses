#include "Utils.h"
#include <GaussianProcess/kernel/SquaredExponential.h>
#include <TrainingTools/iterative/solvers/GradientDescend.h>
#include <iostream>

const std::function<Eigen::VectorXd(const Eigen::VectorXd &)>
    function_to_approximate = [](const Eigen::VectorXd &point) {
      Eigen::VectorXd result(1);
      result << -5.0 + 0.01 * pow(point(0), 4.0) + 0.25 * cos(point(0));
      return result;
    };

std::vector<LogPrediction>
get_predictions(const std::vector<Eigen::VectorXd> &input_samples,
                const gauss::gp::GaussianProcess &process);

int main() {
  const std::size_t samples_in_train_set = 40;
  const std::size_t samples_for_prediction = 40; // 100;
  const double interval_min = -2.0;
  const double interval_max = 2.0;

  // generate samples from the real function
  auto input_space_samples =
      get_equispaced_samples(interval_min, interval_max, samples_in_train_set);
  auto output_space_samples =
      get_output_samples(input_space_samples, function_to_approximate);
  std::cout << "samples generated" << std::endl;

  // generate the approximating gaussian process
  gauss::gp::GaussianProcess process(
      std::make_unique<gauss::gp::SquaredExponential>(1, 0.05),
      gauss::gp::TrainSet{input_space_samples, output_space_samples});
  std::cout << "Gaussian process generated" << std::endl;

  // generate the predictions
  auto predictions_input = get_equispaced_samples(interval_min, interval_max,
                                                  samples_for_prediction);
  auto predictions = get_predictions(predictions_input, process);
  std::cout << "predictions generated" << std::endl;

  // log the predictions
  log_predictions(predictions_input, function_to_approximate, predictions,
                  "predictions_1d.txt");

  //   // tune parameters to get better predictions
  //   train::GradientDescend{}.train(process);
  //   std::cout << "tuning of parameters done" << std::endl;

  //   // generate the predictions with tuned model
  //   predictions = get_predictions(predictions_input, process);
  //   std::cout << "predictions generated again" << std::endl;

  //   // log new predictions
  //   log_predictions(predictions_input, function_to_approximate, predictions,
  //                   "predictions_1d_tuned.txt");

  std::cout
      << "Launch the python script Visualize-1d.py to visualize the results"
      << std::endl;

  return EXIT_SUCCESS;
}

std::vector<LogPrediction>
get_predictions(const std::vector<Eigen::VectorXd> &input_samples,
                const gauss::gp::GaussianProcess &process) {
  std::vector<LogPrediction> predictions;
  predictions.reserve(input_samples.size());
  for (const auto &sample : input_samples) {
    auto prediction = process.predict2(sample);
    Eigen::VectorXd temp(1);
    temp << prediction.mean;
    predictions.push_back(LogPrediction{temp, prediction.covariance});
  }
  return predictions;
}