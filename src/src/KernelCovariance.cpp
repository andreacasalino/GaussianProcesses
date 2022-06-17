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
}

KernelCovariance::~KernelCovariance() = default;

Eigen::MatrixXd KernelCovariance::getCovariance() const {
  return getKernelMatrix();
}

Eigen::MatrixXd KernelCovariance::getCovarianceInv() const {
  return getKernelMatrixInverse();
}

double KernelCovariance::getCovarianceDeterminant() const {
  double result = 1.0;
  for (const auto &eig_val :
       accessKernelRepresentation().kernel_matrix_decomposition.eigenValues) {
    result *= eig_val;
  }
  return result;
}

KernelCovariance::Decomposition
KernelCovariance::getCovarianceDecomposition() const {
  return getKernelMatrixDecomposition();
}

void KernelCovariance::updateKernelFunction(KernelFunctionPtr new_kernel) {
  if (nullptr == new_kernel) {
    throw Error{"Null kernel function"};
  }
  kernelFunction = std::move(new_kernel);
  resetKernelMatrix();
}

void KernelCovariance::resetKernelMatrix() { kernel_representation.reset(); }

const Eigen::MatrixXd &KernelCovariance::getKernelMatrix() const {
  return accessKernelRepresentation().kernel_matrix->access();
}

const Eigen::MatrixXd &KernelCovariance::getKernelMatrixInverse() const {
  return accessKernelRepresentation().kernel_matrix_inverse;
}

const KernelCovariance::Decomposition &
KernelCovariance::getKernelMatrixDecomposition() const {
  return accessKernelRepresentation().kernel_matrix_decomposition;
}

const KernelCovariance::KernelRepresentation &
KernelCovariance::accessKernelRepresentation() const {
  if (nullptr == kernel_representation) {
    kernel_representation = std::make_unique<KernelRepresentation>();
    kernel_representation->kernel_matrix =
        std::make_unique<SymmetricResizableMatrix>(
            [this](const Eigen::Index row, const Eigen::Index col) {
              const auto samples = this->getTrainSet().GetSamplesInput();
              auto result = this->kernelFunction->evaluate(
                  samples[static_cast<std::size_t>(row)],
                  samples[static_cast<std::size_t>(col)]);
              if (row == col) {
                result += this->white_noise_cov;
              }
              return result;
            });
  }
  kernel_representation->kernel_matrix->resize(
      getTrainSet().GetSamplesInput().size());
  if (kernel_representation->kernel_matrix->getComputedSize() !=
      kernel_representation->kernel_matrix->getSize()) {
    // decompose kernel
    const auto &kernel_mat = kernel_representation->kernel_matrix->access();
    auto &decomposition = kernel_representation->kernel_matrix_decomposition;
    Eigen::EigenSolver<Eigen::MatrixXd> solver(kernel_mat);
    decomposition.eigenVectors = solver.eigenvectors().real();
    for (Eigen::Index c = 0; c < decomposition.eigenVectors.cols(); ++c) {
      decomposition.eigenVectors.col(c) /=
          decomposition.eigenVectors.col(c).norm();
    }
    decomposition.eigenValues = solver.eigenvalues().real();
    decomposition.eigenValues_inv = decomposition.eigenValues;
    for (auto &val : decomposition.eigenValues_inv) {
      val = 1.0 / val;
    }
    for (const auto &eig : decomposition.eigenValues) {
      if (eig < SuspiciousCovarianceError::COVARIANCE_TOLLERANCE) {
        throw SuspiciousCovarianceError{
            "Kernel covariance is not positive definite"};
      }
    }
    // compute inverse
    kernel_representation->kernel_matrix_inverse =
        Eigen::MatrixXd(decomposition.eigenVectors *
                        decomposition.eigenValues_inv.asDiagonal() *
                        decomposition.eigenVectors.transpose());
  }
  return *kernel_representation;
}
} // namespace gauss::gp
