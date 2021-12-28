#include "Utils.h"
#include <GaussianProcess/kernel/SquaredExponential.h>
#include <TrainingTools/iterative/solvers/GradientDescend.h>
#include <iostream>

const std::function<double(const Eigen::VectorXd &)> function_to_approximate =
    [](const Eigen::VectorXd &point) { return cos(point.norm()); };

std::vector<Eigen::VectorXd> get_input_samples(const double ray,
                                               const std::size_t samples);

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_grid(const double ray, const std::size_t samples_along_axis);

int main() {
  const std::size_t samples_in_train_set = 20;
  const std::size_t samples_for_preditive_grid = 25;
  const double ray = 6.3;

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
      std::make_unique<gauss::gp::SquaredExponential>(1.0, 0.05),
      gauss::gp::TrainSet{input_samples, convert(output_samples)});
  std::cout << "Gaussian process generated" << std::endl;

  // generate the predictions
  std::vector<std::vector<double>> input_x_coord_predictions,
      input_y_coord_predictions;
  {
    auto grid = get_grid(ray, samples_for_preditive_grid);
    input_x_coord_predictions = std::move(grid.first);
    input_x_coord_predictions = std::move(grid.second);
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

  // // tune parameters to get better predictions
  // train::GradientDescend{}.train(process);
  // std::cout << "tuning of parameters done" << std::endl;

  // re-generate the predictions with tuned model
  prediction_means.clear();
  prediction_covariances.clear();
  for (std::size_t r = 0; r < samples_for_preditive_grid; ++r) {
    prediction_means.emplace_back();
    prediction_covariances.emplace_back();
    for (std::size_t c = 0; c < samples_for_preditive_grid; ++c) {
      Eigen::VectorXd point(2);
      point << input_x_coord_predictions[r][c], input_y_coord_predictions[r][c];
      auto temp = process.predict2(point);
      prediction_means.back().push_back(temp.mean);
      prediction_covariances.back().push_back(temp.covariance);
    }
  }
  std::cout << "predictions generated again" << std::endl;

  // log new predictions
  {
    LoggerExtra logger;
    logger.add_samples(input_samples, output_samples);
    logger.add_field(input_x_coord_predictions, "input_x_coord_predictions");
    logger.add_field(input_y_coord_predictions, "input_y_coord_predictions");
    logger.add_field(output_predictions, "output_predictions");
    logger.add_field(prediction_means, "prediction_means");
    logger.add_field(prediction_covariances, "prediction_covariances");
    logger.print("predictions_after_tune.json");
  }

  std::cout << "call 'python Visualize02.py' to visualize the results"
            << std::endl;

  return EXIT_SUCCESS;
}

std::vector<Eigen::VectorXd> get_input_samples(const double ray,
                                               const std::size_t samples) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples);
  for (std::size_t s = 0; s < samples; ++s) {
    result.emplace_back(2);
    result.back().setRandom();
    result.back() *= ray;
  }
  return result;
};

std::pair<std::vector<std::vector<double>>, std::vector<std::vector<double>>>
get_grid(const double ray, const std::size_t samples_along_axis) {
  std::vector<std::vector<double>> x_coord, y_coord;
  x_coord.reserve(samples_along_axis);
  y_coord.reserve(samples_along_axis);
  auto x_samples = get_equispaced_samples(-ray, ray, samples_along_axis);
  auto y_samples = get_equispaced_samples(-ray, ray, samples_along_axis);
  for (std::size_t r = 0; r < samples_along_axis; ++r) {
    x_coord.emplace_back();
    x_coord.back().reserve(samples_along_axis);
    y_coord.emplace_back();
    y_coord.back().reserve(samples_along_axis);
    for (std::size_t c = 0; c < samples_along_axis; ++c) {
      x_coord.back().push_back(x_samples[r]);
      y_coord.back().push_back(y_samples[c]);
    }
  }
  return std::make_pair<std::vector<std::vector<double>>,
                        std::vector<std::vector<double>>>(std::move(x_coord),
                                                          std::move(y_coord));
}
