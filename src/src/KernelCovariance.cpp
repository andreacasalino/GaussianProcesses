/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <Common.h>
#include <Eigen/Dense>
#include <GaussianProcess/Error.h>
#include <GaussianProcess/KernelCovariance.h>
#include <GaussianUtils/Utils.h>

namespace gauss::gp {
KernelCovariance::KernelCovariance(KernelFunctionPtr new_kernel)
    : kernelFunction(std::move(new_kernel)) {
  if (nullptr == kernelFunction) {
    throw Error{"Found null kernel function"};
  }
  kernel_matrix = std::make_unique<SymmetricResizableMatrix>(
      [this](const Eigen::Index row, const Eigen::Index col) {
        const auto samples = this->getTrainSet().GetSamplesInput();
        return this->kernelFunction->evaluate(
            samples[static_cast<std::size_t>(row)],
            samples[static_cast<std::size_t>(col)]);
      });
}

KernelCovariance::~KernelCovariance() = default;

const Eigen::MatrixXd &KernelCovariance::getKernelMatrix() const {
  kernel_matrix->resize(getTrainSet().GetSamplesInput().size());
  return kernel_matrix->access();
}

const Eigen::MatrixXd &KernelCovariance::getKernelMatrixInverse() const {
  const auto &kernel_mat = getKernelMatrix();
  if ((nullptr == last_kernel_mat_4_kernel_matrix_inverse) ||
      (&kernel_mat != last_kernel_mat_4_kernel_matrix_inverse)) {
    const auto &decomposition = getKernelMatrixDecomposition();
    kernel_matrix_inverse =
        Eigen::MatrixXd(decomposition.eigenVectors *
                        decomposition.eigenValues_inv.asDiagonal() *
                        decomposition.eigenVectors.transpose());
    last_kernel_mat_4_kernel_matrix_inverse = &kernel_mat;
  }
  return kernel_matrix_inverse;
}

const KernelCovariance::Decomposition &
KernelCovariance::getKernelMatrixDecomposition() const {
  const auto &kernel_mat = getKernelMatrix();
  if ((nullptr == last_kernel_mat_4_kernel_matrix_decomposition) ||
      (&kernel_mat != last_kernel_mat_4_kernel_matrix_decomposition)) {
    Eigen::EigenSolver<Eigen::MatrixXd> solver(kernel_mat);
    kernel_matrix_decomposition.eigenVectors = solver.eigenvectors().real();
    for (Eigen::Index c = 0;
         c < kernel_matrix_decomposition.eigenVectors.cols(); ++c) {
      kernel_matrix_decomposition.eigenVectors.col(c) /=
          kernel_matrix_decomposition.eigenVectors.col(c).norm();
    }
    kernel_matrix_decomposition.eigenValues = solver.eigenvalues().real();
    kernel_matrix_decomposition.eigenValues_inv =
        kernel_matrix_decomposition.eigenValues;
    for (auto &val : kernel_matrix_decomposition.eigenValues_inv) {
      val = 1.0 / val;
    }
    last_kernel_mat_4_kernel_matrix_decomposition = &kernel_mat;
  }
  return kernel_matrix_decomposition;
}

double KernelCovariance::getCovarianceDeterminant() const {
  const auto &decomposition = getKernelMatrixDecomposition();
  double result = 1.0;
  for (const auto &eig_val : decomposition.eigenValues) {
    result *= eig_val;
  }
  return result;
}

void KernelCovariance::updateKernelFuction(KernelFunctionPtr new_kernel) {
  kernelFunction = std::move(new_kernel);
  resetKernelMatrix();
}

void KernelCovariance::resetKernelMatrix() { kernel_matrix->resize(0); }
} // namespace gauss::gp
