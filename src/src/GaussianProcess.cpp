/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <GaussianProcess/Error.h>
#include <GaussianProcess/GaussianProcess.h>

#include <TrainingTools/ParametersAware.h>

#include "Common.h"

// #include <fstream>

namespace gauss::gp {
Eigen::VectorXd GaussianProcess::getKx(const Eigen::VectorXd &point) const {
  const auto &input_samples = getTrainSet().GetSamplesInput();
  Eigen::VectorXd Kx(input_samples.size());
  Eigen::Index pos = 0;
  const auto &kernel_function = getKernelFunction();
  for (const auto &sample : input_samples) {
    Kx(pos) = kernel_function.evaluate(point, sample);
    ++pos;
  }
  return Kx;
}

Eigen::VectorXd
GaussianProcess::predict(const Eigen::VectorXd &point, double &covariance,
                         const bool accept_bad_covariance) const {
  Eigen::VectorXd Kx = getKx(point);
  const Eigen::MatrixXd &K_inverse = getKernelMatrixInverse();
  covariance = getKernelFunction().evaluate(point, point);
  const Eigen::MatrixXd covariance_mat = Kx.transpose() * K_inverse * Kx;
  covariance -= covariance_mat(0, 0);
  if ((covariance < SuspiciousCovarianceError::COVARIANCE_TOLLERANCE) &&
      (!accept_bad_covariance)) {
    throw SuspiciousCovarianceError{"Negative covariance for prediction"};
  }
  covariance = std::abs(covariance);
  return getYYpredict_() * K_inverse * Kx;
}

Eigen::VectorXd GaussianProcess::getHyperParameters() const {
  const auto &params = getKernelFunction().getParameters();
  Eigen::VectorXd result(params.size());
  Eigen::Index i = 0;
  for (const auto &parameter : params) {
    result(i) = parameter;
    ++i;
  }
  return result;
}

void GaussianProcess::setHyperParameters(const Eigen::VectorXd &parameters) {
  auto &kernel_function = getKernelFunction();
  if (static_cast<std::size_t>(parameters.size()) !=
      kernel_function.numberOfParameters()) {
    throw gauss::gp::Error("Invalid new set of parameters");
  }
  Eigen::Index i = 0;
  std::vector<double> params_vec;
  params_vec.reserve(static_cast<std::size_t>(parameters.size()));
  for (auto &parameter : parameters) {
    params_vec.push_back(parameters(i));
    ++i;
  }
  getKernelFunction_().setParameters(params_vec);
  resetKernelMatrix();
}

double GaussianProcess::getLogLikelihood() const {
  double result = 0;
  const auto coeff =
      0.5 * static_cast<double>(getTrainSet().getOutputStateSpaceSize());
  for (const auto &eig_val : getCovarianceDecomposition().eigenValues) {
    result -= coeff * log(eig_val);
  }
  result -= 0.5 * trace_product(getKernelMatrixInverse(), getYYtrain_());
  return result;
}

Eigen::VectorXd GaussianProcess::getHyperParametersGradient() const {
  const auto parameters_numb = getKernelFunction().numberOfParameters();
  const auto &samples_in = getTrainSet().GetSamplesInput();

  std::vector<Eigen::MatrixXd> kernel_matrix_gradients;
  {
    kernel_matrix_gradients.reserve(parameters_numb);
    for (std::size_t p = 0; p < parameters_numb; ++p) {
      kernel_matrix_gradients.emplace_back(
          static_cast<Eigen::Index>(samples_in.size()),
          static_cast<Eigen::Index>(samples_in.size()));
    }
    auto fill_kernel_matrix_gradients_at =
        [&kernel_function = getKernelFunction(), &samples_in,
         &kernel_matrix_gradients](const std::size_t r, const std::size_t c) {
          auto grad = kernel_function.getGradient(samples_in[r], samples_in[c]);
          for (std::size_t k = 0; k < grad.size(); ++k) {
            kernel_matrix_gradients[k](r, c) = grad[k];
            if (r != c) {
              kernel_matrix_gradients[k](c, r) =
                  kernel_matrix_gradients[k](r, c);
            }
          }
        };
    for (std::size_t r = 0; r < samples_in.size(); ++r) {
      for (std::size_t c = r; c < samples_in.size(); ++c) {
        fill_kernel_matrix_gradients_at(r, c);
      }
    }
  }

  // {
  //   std::ofstream stream("kernel_gradients");
  //   for (const auto &kernel_matrix_gradient : kernel_matrix_gradients) {
  //     stream << kernel_matrix_gradient << std::endl;
  //   }
  // }

  const auto &YY = getYYtrain_();
  Eigen::VectorXd result(static_cast<Eigen::Index>(parameters_numb));
  const Eigen::MatrixXd &kernel_inv = getKernelMatrixInverse();
  // {
  //   std::ofstream stream("kernel");
  //   stream << getKernelMatrix() << std::endl;
  // }
  // {
  //   std::ofstream stream("kernel_inverse");
  //   stream << kernel_inv << std::endl;
  // }
  Eigen::MatrixXd B =
      kernel_inv * YY -
      Eigen::MatrixXd::Identity(YY.rows(), YY.rows()) *
          static_cast<double>(getTrainSet().getOutputStateSpaceSize());
  for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(parameters_numb);
       ++i) {
    const auto &gradient_matrix =
        kernel_matrix_gradients[static_cast<std::size_t>(i)];
    Eigen::MatrixXd A = kernel_inv * gradient_matrix;
    result(i) = trace_product(A, B);
    result(i) *= 0.5;
  }
  return result;
};

namespace {
class GaussianProcessTrainWrapper : public ::train::ParametersAware {
public:
  GaussianProcessTrainWrapper(
      GaussianProcess &subject,
      const std::optional<gauss::GaussianDistribution> &hyperparameters_prior)
      : subject(subject) {
    if (std::nullopt != hyperparameters_prior) {
      if (subject.getKernelFunction().numberOfParameters() !=
          hyperparameters_prior->getStateSpaceSize()) {
        throw Error{"Invalid size for the gaussian distribution describing "
                    "prior knowledge of hyperparameters"};
      }
      auto &prior = prior_distribution.emplace();
      prior.cov_inv = hyperparameters_prior->getCovarianceInv();
      prior.mean = hyperparameters_prior->getMean();
    }
  }

  ::train::Vect getParameters() const final {
    return subject.getHyperParameters();
  }
  void setParameters(const ::train::Vect &parameters) final {
    subject.setHyperParameters(parameters);
  }
  ::train::Vect getGradient() const final {
    auto result = subject.getHyperParametersGradient();
    if (std::nullopt != prior_distribution) {
      result += prior_distribution->cov_inv *
                (subject.getHyperParameters() - prior_distribution->mean);
    }
    return result;
  };

private:
  GaussianProcess &subject;

  struct CovInvAndMean {
    Eigen::VectorXd mean;
    Eigen::MatrixXd cov_inv;
  };
  std::optional<CovInvAndMean> prior_distribution;
};
} // namespace

void train(
    GaussianProcess &subject, ::train::Trainer &trainer,
    const std::optional<gauss::GaussianDistribution> &hyperparameters_prior) {
  GaussianProcessTrainWrapper wrapper(subject, hyperparameters_prior);
  trainer.maximize();
  trainer.train(wrapper);
}

std::vector<gauss::GaussianDistribution>
GaussianProcess::predict(const Eigen::VectorXd &point,
                         const bool accept_bad_covariance) const {
  double prediction_covariance;
  Eigen::VectorXd prediction_mean = GaussianProcess::predict(
      point, prediction_covariance, accept_bad_covariance);
  Eigen::MatrixXd prediction_covariance_mat(1, 1);
  prediction_covariance_mat << prediction_covariance;
  std::vector<gauss::GaussianDistribution> result;
  result.reserve(prediction_mean.size());
  for (Eigen::Index k = 0; k < prediction_mean.size(); ++k) {
    Eigen::VectorXd temp(1);
    temp << prediction_mean(k);
    result.emplace_back(temp, prediction_covariance_mat);
  }
  return result;
}

GaussianProcess::Prediction
GaussianProcess::predict2(const Eigen::VectorXd &point,
                          const bool accept_bad_covariance) const {
  double covariance;
  Eigen::VectorXd mean =
      GaussianProcess::predict(point, covariance, accept_bad_covariance);
  return Prediction{mean, covariance};
}

GaussianDistribution
GaussianProcess::predict3(const Eigen::VectorXd &point,
                          const bool accept_bad_covariance) const {
  double covariance;
  Eigen::VectorXd mean =
      GaussianProcess::predict(point, covariance, accept_bad_covariance);
  Eigen::MatrixXd big_cov(mean.size(), mean.size());
  big_cov.setZero();
  for (Eigen::Index i = 0; i < big_cov.size(); ++i) {
    big_cov(i, i) = covariance;
  }
  return GaussianDistribution{mean, big_cov};
}
} // namespace gauss::gp
