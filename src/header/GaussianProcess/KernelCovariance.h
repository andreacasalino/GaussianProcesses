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
class SymmetricResizableMatrix;

class KernelCovariance : virtual public TrainSetAware, public CovarianceAware {
public:
  ~KernelCovariance();

  /**
   * @return The kernel of the process.
   * The kernel is lazy computed, as is assumed equal to null before
   * actually adding samples to the process.
   * @throw In case the kernel was not computed as no samples are available
   */
  Eigen::MatrixXd getCovariance() const override;
  /**
   * @return The inverse of the kernel of the process
   * The kernel is lazy computed, as is assumed equal to null before actually
   * adding samples to the process.
   * @throw In case the kernel was not computed as no samples are available
   */
  Eigen::MatrixXd getCovarianceInv() const override;
  /**
   * @return The determinant of the kernel of the process
   * The kernel is lazy computed, as is assumed equal to null before actually
   * adding samples to the process.
   * @throw In case the kernel was not computed as no samples are available
   */
  double getCovarianceDeterminant() const override;

  struct Decomposition {
    Eigen::MatrixXd eigenVectors;
    Eigen::VectorXd eigenValues;
    Eigen::VectorXd eigenValues_inv;
  };
  Decomposition getCovarianceDecomposition() const;

  /**
   * @brief replace the kernel function.
   *
   * @param new_kernel
   * @throw passing a null kernel
   */
  void updateKernelFunction(KernelFunctionPtr new_kernel);

  /**
   * @return The kernel function used to compute the kernel of the process
   */
  const KernelFunction &getKernelFunction() const { return *kernelFunction; }

  double getWhiteNoiseCovariance() const { return white_noise_cov; }
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
