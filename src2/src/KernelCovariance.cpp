/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <Common.h>
#include <Eigen/Dense>
#include <GaussianProcess/Error.h>
#include <GaussianProcess/components/KernelMatrix.h>
#include <GaussianUtils/Utils.h>

namespace gauss::gp {
KernelMatrix::~KernelMatrix() = default;

const Eigen::MatrixXd &KernelMatrix::getKernelMatrix() const {
  return kernel_matrix->access();
}

const Eigen::MatrixXd &KernelMatrix::getKernelMatrixInverse() const {
  if (nullptr == kernel_matrix_inverse) {
    const auto &decomposition = getKernelMatrixDecomposition();
    kernel_matrix_inverse = std::make_unique<Eigen::MatrixXd>(
        decomposition.eigenVectors *
        decomposition.eigenValues_inv.asDiagonal() *
        decomposition.eigenVectors.transpose());
  }
  return *kernel_matrix_inverse;
}

const KernelMatrix::Decomposition &
KernelMatrix::getKernelMatrixDecomposition() const {
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

void KernelMatrix::updateKernelMatrix() {
  kernel_matrix->resize(getTrainSet()->GetSamplesInput().size());
}

void KernelMatrix::resetKernelMatrix() {
  kernel_matrix = std::make_unique<SymmetricMatrixExpandable>(
      [this](const Eigen::Index row, const Eigen::Index col) {
        const auto samples = this->getTrainSet()->GetSamplesInput();
        return this->kernelFunction->evaluate(
            samples[static_cast<std::size_t>(row)],
            samples[static_cast<std::size_t>(col)]);
      });
  kernel_matrix->resize(getTrainSet()->GetSamplesInput().size());
}

void KernelMatrix::updateKernelFuction(KernelFunctionPtr new_kernel) {
  kernelFunction = std::move(new_kernel);
  void resetKernelMatrix();
}

KernelMatrix::KernelMatrix(KernelFunctionPtr new_kernel) {
  if (nullptr == new_kernel) {
    throw Error("empty kernel function");
  }
  updateKernelFuction(std::move(new_kernel));
}

double KernelMatrix::getCovarianceDeterminant() const {
  const auto &decomposition = getKernelMatrixDecomposition();
  double result = 1.0;
  for (const auto &eig_val : decomposition.eigenValues) {
    result *= eig_val;
  }
  return result;
}
} // namespace gauss::gp

/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <Common.h>
#include <GaussianProcess/Error.h>
#include <GaussianProcess/components/OutputMatrix.h>

namespace gauss::gp {
OutputMatrix::~OutputMatrix() = default;

const Eigen::MatrixXd &OutputMatrix::getOutputMatrix() const {
  return output_matrix->access();
}

void OutputMatrix::updateOutputMatrix() {
  output_matrix->resize(getTrainSet()->GetSamplesInput().size());
}

void OutputMatrix::resetOutputMatrix() {
  output_matrix = std::make_unique<SymmetricMatrixExpandable>(
      [this](const Eigen::Index row, const Eigen::Index col) {
        const auto samples = this->getTrainSet()->GetSamplesOutput();
        const auto &y1 = samples[static_cast<std::size_t>(row)];
        const auto &y2 = samples[static_cast<std::size_t>(col)];
        return y1.dot(y2);
      });
}

OutputMatrix::OutputMatrix() { resetOutputMatrix(); }
} // namespace gauss::gp
