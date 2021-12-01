/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#pragma once

#include <GaussianProcess/kernel/KernelFunction.h>
#include <GaussianProcess/components/TrainSetAware.h>

namespace gauss::gp {
class KernelAware : virtual public TrainSetAware {
protected:
    KernelAware(const KernelAware&);
    KernelAware& operator=(const KernelAware&);

    KernelAware(KernelAware&&);
    KernelAware& operator=(KernelAware&&);

  KernelAware(KernelFunctionPtr new_kernel);

  void updateKernel();
  void resetKernel();
  const Eigen::MatrixXd &getKernel() const { return *kernel; };
  const Eigen::MatrixXd &getKernelInverse() const { return *kernel_inverse; };

  KernelFunctionPtr kernelFunction;

private:
  std::unique_ptr<const Eigen::MatrixXd> kernel;
  std::unique_ptr<const Eigen::MatrixXd> kernel_inverse;
};
} // namespace gauss::gp
