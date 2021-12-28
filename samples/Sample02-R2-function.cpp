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

int main() {
  const std::size_t samples_in_train_set = 100;
  const std::size_t prediction_grid_resolution = 50;
  const double ray = 2.0;

  // generate samples from the real function
  auto input_samples = get_input_samples(ray, samples_in_train_set);

  auto output_samples =
      get_output_samples(input_samples, function_to_approximate);
  std::cout << "samples generated" << std::endl;

  // generate the approximating gaussian process
  gauss::gp::GaussianProcess process(
      std::make_unique<gauss::gp::SquaredExponential>(0.1, 0.01),
      gauss::gp::TrainSet{input_samples, output_samples});
  std::cout << "Gaussian process generated" << std::endl;

  // generate the predictions
  auto predictions_grid =
      get_equispaced_grid_samples(ray, prediction_grid_resolution);
  auto predictions = get_predictions(predictions_grid, process);
  std::cout << "predictions generated" << std::endl;

  // log the predictions
  {
    Logger logger;
    logger.log_samples(input_samples, output_samples);
    logger.log_predictions(predictions_grid, predictions);
    logger.print("Sample02.json");
  }

  // tune parameters to get better predictions
  train::GradientDescend{}.train(process);
  std::cout << "tuning of parameters done" << std::endl;

  // re-generate the predictions with tuned model
  predictions = get_predictions(predictions_grid, process);
  std::cout << "predictions generated again" << std::endl;

  // log new predictions
  {
    Logger logger;
    logger.log_samples(input_samples, output_samples);
    logger.log_predictions(predictions_grid, predictions);
    logger.print("Sample02-bis.json");
  }

  std::cout << "call 'python Visualize.py Sample02' to visualize the results"
            << std::endl;

  return EXIT_SUCCESS;
}
