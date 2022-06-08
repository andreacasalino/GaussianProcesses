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

std::vector<Eigen::VectorXd> make_samples(const std::size_t samples_numb,
                                          const Eigen::Index sample_size);

static constexpr double TOLL = 1e-4;

bool is_zeros(const Eigen::MatrixXd &subject);

bool is_equal(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b);

bool is_equal_vec(const Eigen::VectorXd &a, const Eigen::VectorXd &b);

bool is_symmetric(const Eigen::MatrixXd &subject);

bool is_inverse(const Eigen::MatrixXd &subject,
                const Eigen::MatrixXd &candidate);
} // namespace gauss::gp::test
