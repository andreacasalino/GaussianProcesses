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
class Kernel {
public:
  Kernel(const Eigen::MatrixXd &k) : kernel(k){};

  const Eigen::MatrixXd &getKernel() const { return kernel; };

  const Eigen::MatrixXd &getKernelInv() const;

  struct Decomposition {
    Eigen::MatrixXd eigenVectors;
    Eigen::VectorXd eigenValues;
    Eigen::VectorXd eigenValues_inv;
  };

  const Decomposition &getDecomposition() const;

private:
  void init() const;

  Eigen::MatrixXd kernel;
  mutable std::unique_ptr<Decomposition> decomposition;
  mutable std::unique_ptr<Eigen::MatrixXd> kernel_inverse;
};

class KernelAware : virtual public TrainSetAware, public CovarianceAware {
public:
  /**
   * @return The kernel of the process.
   * The kernel is lazy computed, as is assumed equal to null before actually
   * adding samples to the process.
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

  /**
   * @return The kernel function used to compute the kernel of the process
   */
  const KernelFunction &getKernelFunction() const { return *kernelFunction; }

protected:
  KernelAware(KernelFunctionPtr new_kernel);

  void updateKernel();
  void resetKernel();

private:
  KernelFunctionPtr kernelFunction;
  std::unique_ptr<Kernel> kernel;
};
} // namespace gauss::gp
