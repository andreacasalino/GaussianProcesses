#include "Utils.h"
#include <GaussianProcess/kernel/SquaredExponential.h>
#include <TrainingTools/iterative/solvers/GradientDescend.h>
#include <iostream>

const std::function<double(const double &)> function_to_approximate =
    [](const double &point) { return sin(point); };

Eigen::VectorXd convert(const double &sample);

std::vector<Eigen::VectorXd> convert(const std::vector<double> &samples);

class Logger {
public:
  void add_field(const std::vector<double> &data, const std::string &name);

  void print(const std::string &file_name) const;

private:
  std::vector<std::string> fields;
};

int main() {
  const std::size_t samples_in_train_set = 6;
  const std::size_t samples_for_prediction = 200;
  const double ray = 3.0;

  // generate samples from the real function
  auto input_samples = get_equispaced_samples(-ray, ray, samples_in_train_set);

  std::vector<double> output_samples;
  output_samples.reserve(input_samples.size());
  for (const auto &input_sample : input_samples) {
    output_samples.push_back(function_to_approximate(input_sample));
  }
  std::cout << "samples generated" << std::endl;

  // generate the approximating gaussian process
  gauss::gp::GaussianProcess process(
      std::make_unique<gauss::gp::SquaredExponential>(1.0, 0.05),
      gauss::gp::TrainSet{convert(input_samples), convert(output_samples)});
  std::cout << "Gaussian process generated" << std::endl;

  // generate the predictions
  auto input_predictions =
      get_equispaced_samples(-ray, ray, samples_for_prediction);
  std::vector<double> output_predictions, prediction_means,
      prediction_covariances;
  output_predictions.reserve(input_predictions.size());
  prediction_means.reserve(input_predictions.size());
  prediction_covariances.reserve(input_predictions.size());
  for (const auto &input_prediction : input_predictions) {
    output_predictions.push_back(function_to_approximate(input_prediction));
    auto temp = process.predict2(convert(input_prediction));
    prediction_means.push_back(temp.mean);
    prediction_covariances.push_back(temp.covariance);
  }
  std::cout << "predictions generated" << std::endl;

  // log the predictions
  {
    Logger logger;
    logger.add_field(input_samples, "input_samples");
    logger.add_field(output_samples, "output_samples");
    logger.add_field(input_predictions, "input_predictions");
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
  for (const auto &input_prediction : input_predictions) {
    auto temp = process.predict2(convert(input_prediction));
    prediction_means.push_back(temp.mean);
    prediction_covariances.push_back(temp.covariance);
  }
  std::cout << "predictions generated again" << std::endl;

  // log new predictions
  {
    Logger logger;
    logger.add_field(input_samples, "input_samples");
    logger.add_field(output_samples, "output_samples");
    logger.add_field(input_predictions, "input_predictions");
    logger.add_field(output_predictions, "output_predictions");
    logger.add_field(prediction_means, "prediction_means");
    logger.add_field(prediction_covariances, "prediction_covariances");
    logger.print("predictions_after_tune.json");
  }

  std::cout << "call 'python Visualize01.py' to visualize the results"
            << std::endl;

  return EXIT_SUCCESS;
}

Eigen::VectorXd convert(const double &sample) {
  Eigen::VectorXd result(1);
  result << sample;
  return result;
}

std::vector<Eigen::VectorXd> convert(const std::vector<double> &samples) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(samples.size());
  for (const auto &sample : samples) {
    result.push_back(convert(sample));
  }
  return result;
}

void Logger::add_field(const std::vector<double> &data,
                       const std::string &name) {
  std::stringstream stream;
  stream << '\"' << name << "\":[";
  auto it = data.begin();
  stream << *it;
  ++it;
  for (it; it != data.end(); ++it) {
    stream << ',' << *it;
  }
  stream << ']';
  fields.push_back(stream.str());
};

void Logger::print(const std::string &file_name) const {
  std::ofstream stream(file_name);
  stream << '{';
  auto it = fields.begin();
  stream << *it;
  ++it;
  for (it; it != fields.end(); ++it) {
    stream << ',' << *it;
  }
  stream << '}';
};
