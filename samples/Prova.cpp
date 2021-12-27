#include <GaussianProcess/GaussianProcess.h>
#include <GaussianProcess/GaussianProcessVectorial.h>
#include <GaussianProcess/kernel/SquaredExponential.h>
#include <iostream>

int main() {
  const std::size_t samples = 10;

  // {
  //   std::vector<Eigen::VectorXd> inputs, outputs;
  //   inputs.reserve(samples);
  //   outputs.reserve(samples);
  //   for (std::size_t s = 0; s < samples; ++s) {
  //     inputs.emplace_back(5);
  //     inputs.back().setRandom();
  //     outputs.emplace_back(3);
  //     outputs.back().setRandom();
  //   }

  //   gauss::gp::GaussianProcessVectorial process(
  //       std::make_unique<gauss::gp::SquaredExponential>(1.0, 0.01),
  //       gauss::gp::TrainSet{inputs, outputs});

  //   auto it_out = outputs.begin();
  //   for (auto it_in = inputs.begin(); it_in != inputs.end();
  //        ++it_in, ++it_out) {
  //     std::cout << it_out->transpose() << std::endl;
  //     std::cout << process.predict2(*it_in).mean.transpose() << std::endl;
  //     Eigen::VectorXd delta(it_in->size());
  //     delta.setRandom();
  //     delta *= 0.05;
  //     std::cout << process.predict2(*it_in + delta).mean.transpose()
  //               << std::endl
  //               << std::endl;
  //   }
  // }

  {
    std::vector<Eigen::VectorXd> inputs, outputs;
    inputs.reserve(samples);
    outputs.reserve(samples);
    for (std::size_t s = 0; s < samples; ++s) {
      inputs.emplace_back(1);
      inputs.back().setRandom();
      outputs.emplace_back(1);
      outputs.back().setRandom();
    }

    gauss::gp::GaussianProcessVectorial process(
        std::make_unique<gauss::gp::SquaredExponential>(1.0, 0.01),
        gauss::gp::TrainSet{inputs, outputs});

    auto it_out = outputs.begin();
    for (auto it_in = inputs.begin(); it_in != inputs.end();
         ++it_in, ++it_out) {
      std::cout << it_out->transpose() << std::endl;
      std::cout << process.predict2(*it_in).mean.transpose() << std::endl;
      Eigen::VectorXd delta(it_in->size());
      delta.setRandom();
      delta *= 0.05;
      std::cout << process.predict2(*it_in + delta).mean.transpose()
                << std::endl
                << std::endl;
    }
  }

  return EXIT_SUCCESS;
}