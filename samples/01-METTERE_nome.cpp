#include <GaussianProcess/GaussianProcessFactory.h>
#include <GaussianProcess/kernel/ExponentialRBF.h>
#include <iostream>

int main() {
  gauss::gp::GaussianProcessFactory generator(
      5, 2, std::make_unique<gauss::gp::ExponentialRBF>(1, 0.05));
  generator.setSamplesNumb(10);
  auto process = generator.makeRandomModel();

  std::cout << process->getCovariance() << std::endl << std::endl << std::endl;
  std::cout << process->getCovarianceInv() << std::endl
            << std::endl
            << std::endl;

  for (const auto &sample :
       process->getTrainSet()->GetSamplesInput().GetSamples()) {
    std::cout << sample.transpose() << std::endl;
  }
  std::cout << std::endl << std::endl;

  auto point = process->getTrainSet()->GetSamplesInput().GetSamples().front();
  {
    Eigen::VectorXd delta(point.size());
    delta.setRandom();
    point += delta;
  }
  auto prediction = process->predict(point);

  return EXIT_SUCCESS;
}