#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/kernel/SquaredExponential.h>

#include <TrainingTools/iterative/solvers/GradientDescend.h>

int main() {
  // build a kernel function
  // for the purpose of this example we will use a simple squared exponential
  gauss::gp::KernelFunctionPtr kernel_function =
      std::make_unique<gauss::gp::SquaredExponential>(1.0, 1.0);

  // build an empty gaussian process absorbing the previously defined kernel
  // function.
  // The input size of the process will be 3, while the output will be 2, i.e.
  // this process will be able to approximate a tri-variate vectorial function
  // made of 2 components.
  gauss::gp::GaussianProcess gauss_process(std::move(kernel_function),
                                           3 // input size
                                           ,
                                           2 // output size
  );

  // fill the training set of the process with samples taken from the (unknown)
  // function to approximate
  std::vector<Eigen::VectorXd> samples =
      ...; // each element will be a 5 sized vector, whose first 3 components
           // represent the values of the input, while the other ones the values
           // of the output
  for (const auto &sample_to_add : samples) {
    gauss_process.getTrainSet().addSample(sample_to_add);
  }

  {
    // predict the output for an input not contained in the training set
    Eigen::VectorXd input_to_predict = ...;
    std::vector<gauss::GaussianDistribution>
        predicted_output // each component is a scalar guassian distribution
                         // with a certain mean and covariance matrix
        = gauss_process.predict(input_to_predict);
  }

  {
    Eigen::VectorXd input_to_predict = ...;
    // access the prediction as a unique multivariate vectorial guassian
    // distribution
    gauss::GaussianDistribution predicted_output =
        gauss_process.predict3(input_to_predict);
  }

  // optimize the hyperparameters through training in order to improve
  // predicting performance.
  ::train::GradientDescendFixed
      gradient_descender; // you can also use another trainer from
  // TrainingTools, or define your own
  gauss::gp::train(gauss_process, gradient_descender);

  return EXIT_SUCCESS;
}
