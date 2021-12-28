#include "Utils.h"
#include <GaussianProcess/kernel/SquaredExponential.h>
#include <TrainingTools/iterative/solvers/GradientDescend.h>
#include <iostream>

const std::function<double(const Eigen::VectorXd &)> function_to_approximate =
    [](const Eigen::VectorXd &point) {
      return 1.0 - 2.0 * point(0) + 0.5 * point(1);
    };

int main() {
  const std::size_t samples_in_train_set = 100;
  const std::size_t samples_for_preditive_grid = 25;
  const double ray = 4.5;

  // generate samples from the real function
  auto input_samples = get_input_samples(ray, samples_in_train_set);

  std::vector<double> output_samples;
  output_samples.reserve(input_samples.size());
  for (const auto &input_sample : input_samples) {
    output_samples.push_back(function_to_approximate(input_sample));
  }
  std::cout << "samples generated" << std::endl;

  // generate the approximating gaussian process
  gauss::gp::GaussianProcess process(
      std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0),
      gauss::gp::TrainSet{input_samples, convert(output_samples)});
  std::cout << "Gaussian process generated" << std::endl;

  // generate the predictions
  std::vector<std::vector<double>> input_x_coord_predictions,
      input_y_coord_predictions;
  {
    auto grid = get_grid(ray, samples_for_preditive_grid);
    input_x_coord_predictions = std::move(grid.first);
    input_y_coord_predictions = std::move(grid.second);
  }
  std::vector<std::vector<double>> output_predictions, prediction_means,
      prediction_covariances;
  for (std::size_t r = 0; r < samples_for_preditive_grid; ++r) {
    output_predictions.emplace_back();
    prediction_means.emplace_back();
    prediction_covariances.emplace_back();
    for (std::size_t c = 0; c < samples_for_preditive_grid; ++c) {
      Eigen::VectorXd point(2);
      point << input_x_coord_predictions[r][c], input_y_coord_predictions[r][c];
      output_predictions.back().push_back(function_to_approximate(point));
      auto temp = process.predict2(point);
      prediction_means.back().push_back(temp.mean);
      prediction_covariances.back().push_back(temp.covariance);
    }
  }
  std::cout << "predictions generated" << std::endl;

  // log the predictions
  {
    LoggerExtra logger;
    logger.add_samples(input_samples, output_samples);
    logger.add_field(input_x_coord_predictions, "input_x_coord_predictions");
    logger.add_field(input_y_coord_predictions, "input_y_coord_predictions");
    logger.add_field(output_predictions, "output_predictions");
    logger.add_field(prediction_means, "prediction_means");
    logger.add_field(prediction_covariances, "prediction_covariances");
    logger.print("predictions.json");
  }

  std::cout << "call 'python Visualize02.py' to visualize the results"
            << std::endl;

  return EXIT_SUCCESS;
}
