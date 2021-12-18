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
class KernelAware : virtual public TrainSetAware, public CovarianceAware {
public:
  Eigen::MatrixXd getCovariance() const override { return *kernel; };
  Eigen::MatrixXd getCovarianceInv() const override { return *kernel_inverse; };
  double getCovarianceDeterminant() const override {
    return *kernel_determinant;
  };

  const KernelFunction &getKernelFunction() const { return *kernelFunction; }

protected:
  KernelAware(KernelFunctionPtr new_kernel);

  void updateKernel();
  void resetKernel();
  const Eigen::MatrixXd &getKernel() const { return *kernel; };
  const Eigen::MatrixXd &getKernelInverse() const { return *kernel_inverse; };

  KernelFunctionPtr kernelFunction;

private:
  std::unique_ptr<const Eigen::MatrixXd> kernel;
  std::unique_ptr<const Eigen::MatrixXd> kernel_inverse;
  std::unique_ptr<double> kernel_determinant;
};
} // namespace gauss::gp
