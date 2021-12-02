/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <Common.h>
#include <Eigen/Dense>
#include <GaussianProcess/Error.h>
#include <GaussianProcess/components/KernelAware.h>
#include <GaussianUtils/Utils.h>

namespace gauss::gp {
KernelAware::KernelAware(KernelFunctionPtr new_kernel) {
  if (nullptr == new_kernel) {
    throw Error("empty kernel function");
  }
  kernelFunction = std::move(new_kernel);
}

namespace {
Eigen::MatrixXd
compute_kernel_portion(const std::vector<Eigen::VectorXd> &samples,
                       const KernelFunction &function,
                       const MatrixIndices &rows, const MatrixIndices &cols) {
  Eigen::MatrixXd result(rows.end - rows.start, cols.end - cols.start);
  std::size_t c;
  for (std::size_t r = rows.start; r < rows.end; ++r) {
    for (c = cols.start; c < cols.end; ++c) {
      result(r, c) = function.evaluate(samples[r], samples[c]);
    }
  }
  return result;
}
} // namespace

void KernelAware::updateKernel() {
  const auto &input_samples = samples->GetSamplesInput().GetSamples();
  if (nullptr == kernel) {
    kernel = std::make_unique<Eigen::MatrixXd>(compute_kernel_portion(
        input_samples, *kernelFunction, MatrixIndices{0, input_samples.size()},
        MatrixIndices{0, input_samples.size()}));
  } else {
    MatrixIndices old_indices = MatrixIndices{0, kernel->rows()};
    MatrixIndices new_indices =
        MatrixIndices{kernel->rows(), input_samples.size()};
    Eigen::MatrixXd K_old_new = compute_kernel_portion(
        input_samples, *kernelFunction, old_indices, new_indices);
    Eigen::MatrixXd K_new_new = compute_kernel_portion(
        input_samples, *kernelFunction, new_indices, new_indices);
    auto new_kernel = std::make_unique<Eigen::MatrixXd>(input_samples.size(),
                                                        input_samples.size());
    set_matrix_portion(*new_kernel, *kernel, old_indices, old_indices);
    set_matrix_portion(*new_kernel, K_old_new, old_indices, new_indices);
    set_matrix_portion(*new_kernel, K_old_new.transpose(), new_indices,
                       old_indices);
    set_matrix_portion(*new_kernel, K_new_new, new_indices, new_indices);
    kernel = std::move(new_kernel);
  }
  kernel_inverse.reset(new Eigen::MatrixXd(computeCovarianceInvert(*kernel)));
}

void KernelAware::resetKernel() {
  kernel.reset();
  kernel_inverse.reset();
}

double KernelAware::getCovarianceDeterminant() const {
  return kernel->determinant();
}
} // namespace gauss::gp
