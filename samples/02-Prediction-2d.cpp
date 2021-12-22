#include "Utils.h"
#include <GaussianProcess/kernel/SquaredExponential.h>
#include <GaussianProcess/train/Trainer.h>
#include <iostream>

const std::function<Eigen::VectorXd(const Eigen::VectorXd &)>
    function_to_approximate = [](const Eigen::VectorXd &point) {
      Eigen::VectorXd result(1);
      const auto &point_val = point.norm();
      result << -5.0 + 0.01 * pow(point_val, 4.0) + 0.25 * cos(point_val);
      return result;
    };

int main() {
  const std::size_t samples_in_train_set = 10;
  const std::size_t samples_for_prediction = 50;
  const double interval_min = -2.0;
  const double interval_max = 2.0;

  // generate samples from the real function
  auto input_space_samples =
      get_input_samples(interval_min, interval_max, interval_min, interval_max,
                        samples_in_train_set);
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
      get_input_samples(interval_min, interval_max, interval_min, interval_max,
                        samples_for_prediction);
  auto predictions = get_predictions(predictions_input, process);
  std::cout << "predictions generated" << std::endl;

  // log the predictions
  log_predictions(predictions_input, predictions, "predictions_2d.txt");

  // tune parameters to get better predictions
  gauss::gp::train(process, 20);
  std::cout << "tuning of parameters done" << std::endl;

  // generate the predictions with tuned model
  predictions = get_predictions(predictions_input, process);
  std::cout << "predictions generated again" << std::endl;

  // log new predictions
  log_predictions(predictions_input, predictions, "predictions_2d_tuned.txt");

  std::cout
      << "Launch the python script Visualize-2d.py to visualize the results"
      << std::endl;

  return EXIT_SUCCESS;
}