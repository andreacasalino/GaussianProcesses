/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/train/Trainer.h>

namespace gauss::gp {
void train(GaussianProcessBase &process, const std::size_t iterations) {
  for (std::size_t k = 0; k < iterations; ++k) {
    auto parameters = process.getParameters();
    auto grad = 0.1 * process.getParametersGradient();
    parameters += grad;
    process.setParameters(parameters);
  }
}
} // namespace gauss::gp
