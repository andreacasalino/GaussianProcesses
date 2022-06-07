/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>

namespace gauss::gp::test {
class TestFunction : public KernelFunction {
public:
  TestFunction(const double teta = 1.0);

  std::unique_ptr<KernelFunction> copy() const override {
    return std::make_unique<TestFunction>(teta);
  }

  std::size_t numberOfParameters() const override { return 1; }

  std::vector<double> getParameters() const override { return {teta}; }

  void setParameters(const std::vector<double> &values) override;

  double evaluate(const Eigen::VectorXd &a,
                  const Eigen::VectorXd &b) const override;

  std::vector<double> getGradient(const Eigen::VectorXd &a,
                                  const Eigen::VectorXd &b) const override;

private:
  double teta;
};
} // namespace gauss::gp::test

// #include <GaussianProcess/components/GaussianProcessBase.h>
// #include <GaussianProcess/kernel/Linear.h>
// #include <gtest/gtest.h>

// namespace gauss::gp::test {
// template <std::size_t InputSize, std::size_t OutputSize>
// class GaussianProcessTest : public GaussianProcessBase, public
// ::testing::Test { public:
//   GaussianProcessTest()
//       : GaussianProcessBase(std::make_unique<LinearFunction>(1, 1,
//       InputSize),
//                             InputSize, OutputSize){};

//   GaussianProcessTest(KernelFunctionPtr kernel_function)
//       : GaussianProcessBase(std::move(kernel_function), InputSize,
//                             OutputSize){};

// protected:
//   Eigen::VectorXd make_sample_input() const {
//     Eigen::VectorXd result(InputSize);
//     result.setRandom();
//     return result;
//   };
//   Eigen::VectorXd make_sample_output() const {
//     Eigen::VectorXd result(OutputSize);
//     result.setRandom();
//     return result;
//   };

//   std::vector<Eigen::VectorXd>
//   make_samples_input(const std::size_t samples_numb) const {
//     std::vector<Eigen::VectorXd> input_samples;
//     input_samples.reserve(samples_numb);
//     for (std::size_t s = 0; s < samples_numb; ++s) {
//       input_samples.emplace_back(make_sample_input());
//     }
//     return input_samples;
//   };

//   std::vector<Eigen::VectorXd>
//   make_samples_output(const std::size_t samples_numb) const {
//     std::vector<Eigen::VectorXd> output_samples;
//     output_samples.reserve(samples_numb);
//     for (std::size_t s = 0; s < samples_numb; ++s) {
//       output_samples.emplace_back(make_sample_output());
//     }
//     return output_samples;
//   };

//   TrainSet make_samples(const std::size_t samples_numb) const {
//     return gauss::gp::TrainSet{make_samples_input(samples_numb),
//                                make_samples_output(samples_numb)};
//   };
// };
// } // namespace gauss::gp::test
