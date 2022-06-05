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
  if (nullptr == kernel_matrix_inverse) {
    const auto &decomposition = getKernelMatrixDecomposition();
    kernel_matrix_inverse = std::make_unique<Eigen::MatrixXd>(
        decomposition.eigenVectors *
        decomposition.eigenValues_inv.asDiagonal() *
        decomposition.eigenVectors.transpose());
  }
  return *kernel_matrix_inverse;
}

const KernelCovariance::Decomposition &
KernelCovariance::getKernelMatrixDecomposition() const {
  if (nullptr == kernel_matrix_decomposition) {
    kernel_matrix_decomposition = std::make_unique<Decomposition>();
    auto &subject = *kernel_matrix_decomposition;
    const auto &matrix = kernel_matrix->access();
    Eigen::EigenSolver<Eigen::MatrixXd> solver(matrix);
    subject.eigenVectors = solver.eigenvectors().real();
    for (Eigen::Index c = 0; c < subject.eigenVectors.cols(); ++c) {
      subject.eigenVectors.col(c) /= subject.eigenVectors.col(c).norm();
    }
    subject.eigenValues = solver.eigenvalues().real();
    subject.eigenValues_inv = subject.eigenValues;
    for (auto &val : subject.eigenValues_inv) {
      val = 1.0 / val;
    }
  }
  return *kernel_matrix_decomposition;
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
