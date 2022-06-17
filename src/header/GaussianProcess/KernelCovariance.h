/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/TrainSet.h>
#include <GaussianProcess/kernel/KernelFunction.h>
#include <GaussianUtils/components/CoviarianceAware.h>

namespace gauss::gp {
class NotPositiveDefiniteKernelCovarianceError : public Error {
public:
  NotPositiveDefiniteKernelCovarianceError();

  static const double TOLLERANCE;
};

class SymmetricResizableMatrix;

class KernelCovariance : virtual public TrainSetAware, public CovarianceAware {
public:
  ~KernelCovariance() override;

  /**
   * @return The covariance matrix characterizing the process.
   * @throw NotPositiveDefiniteKernelCovarianceError in case the kernel matrix
   * is ill. This is typically due to a bad defined kernel function.
   */
  Eigen::MatrixXd getCovariance() const override;
  /**
   * @return The inverse of the covariance matrix characterizing the process.
   * @throw NotPositiveDefiniteKernelCovarianceError in case the kernel matrix
   * is ill. This is typically due to a bad defined kernel function.
   */
  Eigen::MatrixXd getCovarianceInv() const override;
  /**
   * @return The determinant of the covariance matrix characterizing the
   * process.
   * @throw NotPositiveDefiniteKernelCovarianceError in case the kernel matrix
   * is ill. This is typically due to a bad defined kernel function.
   */
  double getCovarianceDeterminant() const override;

  struct Decomposition {
    Eigen::MatrixXd eigenVectors;
    Eigen::VectorXd eigenValues;
    Eigen::VectorXd eigenValues_inv;
  };
  /**
   * @return The decomposition of the covariance matrix characterizing the
   * process.
   * @throw NotPositiveDefiniteKernelCovarianceError in case the kernel matrix
   * is ill. This is typically due to a bad defined kernel function.
   */
  Decomposition getCovarianceDecomposition() const;

  /**
   * @brief replace the kernel function.
   *
   * @param new_kernel the new kernel function to use for describing the
   * covariance of the process
   * @throw passing a null kernel
   */
  void updateKernelFunction(KernelFunctionPtr new_kernel);

  /**
   * @return The current abosrbed kernel function
   */
  const KernelFunction &getKernelFunction() const { return *kernelFunction; }

  /**
   * @return The white noise covariance contribution. This value should be tuned
   * according to the noise level in the training set. Refer also to the pdf
   * documentation
   */
  double getWhiteNoiseCovariance() const { return white_noise_cov; }
  /**
   * @brief Sets the white noise covariance contribution. This value should be
   * tuned according to the noise level in the training set. Refer also to the
   * pdf documentation
   */
  void setWhiteNoiseStandardDeviation(const double val) {
    white_noise_cov = val * val;
  }

protected:
  KernelCovariance(KernelFunctionPtr new_kernel);

  const Eigen::MatrixXd &getKernelMatrix() const;
  const Eigen::MatrixXd &getKernelMatrixInverse() const;
  const Decomposition &getKernelMatrixDecomposition() const;

  void resetKernelMatrix();

  KernelFunction &getKernelFunction_() { return *kernelFunction; }

private:
  KernelFunctionPtr kernelFunction;

  struct KernelRepresentation {
    std::unique_ptr<SymmetricResizableMatrix> kernel_matrix;
    Decomposition kernel_matrix_decomposition;
    Eigen::MatrixXd kernel_matrix_inverse;
  };
  const KernelRepresentation &accessKernelRepresentation() const;
  mutable std::unique_ptr<KernelRepresentation> kernel_representation;

  double white_noise_cov = 0.1;
};
} // namespace gauss::gp
