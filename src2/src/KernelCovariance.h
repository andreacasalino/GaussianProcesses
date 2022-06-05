/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/components/TrainSetAware.h>
#include <GaussianProcess/kernel/KernelFunction.h>
#include <GaussianUtils/components/CoviarianceAware.h>

namespace gauss::gp {
class SymmetricMatrixExpandable;

class KernelMatrix : virtual public TrainSetAware, public CovarianceAware {
public:
  ~KernelMatrix();

  /**
   * @return The kernel of the process.
   * The kernel is lazy computed, as is assumed equal to null before
   * actually adding samples to the process.
   * @throw In case the kernel was not computed as no samples are available
   */
  Eigen::MatrixXd getCovariance() const override { return getKernelMatrix(); }
  /**
   * @return The inverse of the kernel of the process
   * The kernel is lazy computed, as is assumed equal to null before actually
   * adding samples to the process.
   * @throw In case the kernel was not computed as no samples are available
   */
  Eigen::MatrixXd getCovarianceInv() const override {
    return getKernelMatrixInverse();
  }
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
   * @return The kernel function used to compute the kernel of the process
   */
  const KernelFunction &getKernelFunction() const { return *kernelFunction; }

protected:
  KernelMatrix(KernelFunctionPtr new_kernel);

  const Eigen::MatrixXd &getKernelMatrix() const;
  const Eigen::MatrixXd &getKernelMatrixInverse() const;
  const Decomposition &getKernelMatrixDecomposition() const;

  void updateKernelFuction(KernelFunctionPtr new_kernel);
  void updateKernelMatrix();
  void resetKernelMatrix();

private:
  KernelFunctionPtr kernelFunction;

  std::unique_ptr<SymmetricMatrixExpandable> kernel_matrix;

  mutable std::unique_ptr<Decomposition> kernel_matrix_decomposition;

  mutable std::unique_ptr<Eigen::MatrixXd> kernel_matrix_inverse;
};
} // namespace gauss::gp

/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/components/TrainSetAware.h>

namespace gauss::gp {
class SymmetricMatrixExpandable;

class OutputMatrix : virtual public TrainSetAware {
public:
  ~OutputMatrix();

protected:
  OutputMatrix();

  const Eigen::MatrixXd &getOutputMatrix() const;

  void updateOutputMatrix();
  void resetOutputMatrix();

private:
  std::unique_ptr<SymmetricMatrixExpandable> output_matrix;
};
} // namespace gauss::gp
