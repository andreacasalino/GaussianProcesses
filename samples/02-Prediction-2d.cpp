#include "Utils.h"
#include <GaussianProcess/kernel/SquaredExponential.h>
#include <TrainingTools/iterative/solvers/GradientDescend.h>
#include <iostream>

const std::function<Eigen::VectorXd(const Eigen::VectorXd &)>
    function_to_approximate = [](const Eigen::VectorXd &point) {
      Eigen::VectorXd result(1);
      const auto point_val = point.norm();
      result << -5.0 + 0.01 * pow(point_val, 4.0) + 0.25 * cos(point_val);
      return result;
    };

std::vector<Eigen::VectorXd>
get_equispaced_samples(const double interval_min_x, const double interval_max_x,
                       const double interval_min_y, const double interval_max_y,
                       const std::size_t points_along_each_axis);

std::vector<LogPrediction>
get_predictions(const std::vector<Eigen::VectorXd> &input_samples,
                const gauss::gp::GaussianProcess &process);

int main() {
  const std::size_t samples_in_train_set = 10;
  const std::size_t samples_for_prediction = 50;
  const double interval_min = -2.0;
  const double interval_max = 2.0;

  // generate samples from the real function
  auto input_space_samples =
      get_equispaced_samples(interval_min, interval_max, interval_min,
                             interval_max, samples_in_train_set);
  auto output_space_samples =
      get_output_samples(input_space_samples, function_to_approximate);
  std::cout << "samples generated" << std::endl;

  // generate the approximating gaussian process
  gauss::gp::GaussianProcess process(
      std::make_unique<gauss::gp::SquaredExponential>(0.1, 0.01),
      gauss::gp::TrainSet{input_space_samples, output_space_samples});
  std::cout << "Gaussian process generated" << std::endl;

  // generate the predictions
  auto predictions_input =
      get_equispaced_samples(interval_min, interval_max, interval_min,
                             interval_max, samples_for_prediction);
  auto predictions = get_predictions(predictions_input, process);
  std::cout << "predictions generated" << std::endl;

  // log the predictions
  log_predictions(predictions_input, function_to_approximate, predictions,
                  "predictions_2d.txt");

  // tune parameters to get better predictions
  train::GradientDescend{}.train(process);
  std::cout << "tuning of parameters done" << std::endl;

  // generate the predictions with tuned model
  predictions = get_predictions(predictions_input, process);
  std::cout << "predictions generated again" << std::endl;

  // log new predictions
  log_predictions(predictions_input, function_to_approximate, predictions,
                  "predictions_2d_tuned.txt");

  std::cout
      << "Launch the python script Visualize-2d.py to visualize the results"
      << std::endl;

  return EXIT_SUCCESS;
}

std::vector<Eigen::VectorXd>
get_equispaced_samples(const double interval_min_x, const double interval_max_x,
                       const double interval_min_y, const double interval_max_y,
                       const std::size_t points_along_each_axis) {
  auto samples_along_x = get_equispaced_samples(interval_min_x, interval_max_x,
                                                points_along_each_axis);
  auto samples_along_y = get_equispaced_samples(interval_min_y, interval_max_y,
                                                points_along_each_axis);
  std::vector<Eigen::VectorXd> result;
  result.reserve(points_along_each_axis * points_along_each_axis);
  for (const auto &sample_y : samples_along_y) {
    for (const auto &sample_x : samples_along_x) {
      Eigen::VectorXd temp(2);
      temp << sample_x(0), sample_y(0);
      result.emplace_back(temp);
    }
  }
  return result;
}

std::vector<LogPrediction>
get_predictions(const std::vector<Eigen::VectorXd> &input_samples,
                const gauss::gp::GaussianProcessVectorial &process) {
  std::vector<LogPrediction> predictions;
  predictions.reserve(input_samples.size());
  for (const auto &sample : input_samples) {
    predictions.push_back(process.predict2(sample));
  }
  return predictions;
}