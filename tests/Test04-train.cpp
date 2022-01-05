#include "Utils.h"
#include <GaussianProcess/kernel/SquaredExponential.h>
#include <TrainingTools/iterative/solvers/GradientDescend.h>
#include <functional>
#include <gtest/gtest.h>

static const std::function<double(const double, const Eigen::Index)>
    underlying_vectorial_function =
        [](const double point, const Eigen::Index index) {
          return 0.1 * point * point + static_cast<double>(index);
        };

namespace gauss::gp::test {
template <std::size_t InputSize, std::size_t OutputSize>
class GaussianProcessTrainTest
    : public GaussianProcessTest<InputSize, OutputSize> {
public:
  GaussianProcessTrainTest()
      : GaussianProcessTest<InputSize, OutputSize>(
            std::make_unique<SquaredExponential>(1, 0.5)){};

  void SetUp() {
    auto samples_input = this->make_samples_input(20);
    auto samples_output = this->make_samples_output(20);
    for (auto &sample_out : samples_output) {
      double radial_distance = sample_out.norm();
      auto delta = sample_out;
      delta *= 0.025;
      for (Eigen::Index i = 0; i < sample_out.size(); ++i) {
        sample_out(i) = underlying_vectorial_function(radial_distance, i);
      }
      sample_out += delta;
    }

    this->pushSamples(samples_input, samples_output);
  }

protected:
  void set_and_check_parameters(const Eigen::VectorXd &parameters) {
    auto initial_kernel = this->getCovariance();
    this->setParameters(parameters);
    EXPECT_EQ(this->getParameters(), parameters);
    EXPECT_FALSE(initial_kernel == this->getCovariance());
  }
};
} // namespace gauss::gp::test

using Process3_1 = gauss::gp::test::GaussianProcessTrainTest<3, 1>;
using Process5_3 = gauss::gp::test::GaussianProcessTrainTest<5, 3>;

TEST_F(Process5_3, set_parameters) {
  {
    Eigen::VectorXd par(2);
    par << 0.1, 0.8;
    set_and_check_parameters(par);
  }

  {
    Eigen::VectorXd par(2);
    par << 0.8, 0.05;
    set_and_check_parameters(par);
  }

  {
    Eigen::VectorXd par(3);
    par.setOnes();
    EXPECT_THROW(this->setParameters(par), gauss::gp::Error);
  }

  {
    Eigen::VectorXd par(1);
    par.setOnes();
    EXPECT_THROW(this->setParameters(par), gauss::gp::Error);
  }
}

TEST_F(Process3_1, gradient_likelihood_computation) {
  auto grad = getParametersGradient();
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NO_THROW(getLogLikelihood());
}
TEST_F(Process5_3, gradient_likelihood_computation) {
  auto grad = getParametersGradient();
  EXPECT_EQ(grad.size(), 2);
  EXPECT_NO_THROW(getLogLikelihood());
}

TEST_F(Process5_3, train) {
  auto initial_likelihood = getLogLikelihood();
  {
    train::GradientDescend solver;
    solver.train(*this);
  }
  auto new_likelihood = getLogLikelihood();
  EXPECT_LE(initial_likelihood, new_likelihood);
}

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
