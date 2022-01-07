#include "Utils.h"
#include <GaussianProcess/kernel/SquaredExponential.h>
// #include <TrainingTools/iterative/solvers/QuasiNewton.h>
#include <TrainingTools/iterative/solvers/GradientDescend.h>
#include <iostream>

const std::function<double(const double &)> function_to_approximate =
    [](const double &point) { return sin(point); };

template <typename SolverT> class Tuner : public SolverT {
public:
  Tuner() = default;

  const std::vector<double> &getLikelihoodEvolution() const {
    return likelihood_evolution;
  }

protected:
  void updateDirection() override {
    const gauss::gp::GaussianProcessBase *process =
        dynamic_cast<const gauss::gp::GaussianProcessBase *>(this->getModel());
    likelihood_evolution.push_back(process->getLogLikelihood());
    this->SolverT::updateDirection();
  };
  void initDirection() override {
    likelihood_evolution.clear();
    this->SolverT::initDirection();
  };

private:
  std::vector<double> likelihood_evolution;
};

int main() {
  const std::size_t samples_in_train_set = 12;
  const std::size_t samples_for_prediction = 200;
  const double ray = 6.0;

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
      std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0),
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

  // tune parameters to get better predictions
  Tuner<train::GradientDescendFixed> tuner;
  process.train(tuner);
  // log the likelihood evolution
  {
    std::ofstream evolution("tune_evolution.txt");
    std::size_t iter = 1;
    for (const auto &l : tuner.getLikelihoodEvolution()) {
      evolution << iter << ' ' << l << std::endl;
      ++iter;
    }
  }
  std::cout << "tuning of parameters done" << std::endl;
  std::cout << "call 'python VisualizeEvolution.py' to visualize the check the "
               "model improvement over the iterations"
            << std::endl;

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
