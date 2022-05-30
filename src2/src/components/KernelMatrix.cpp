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
const Eigen::MatrixXd &KernelMatrix::getKernelInv() const {
  if (nullptr == kernel_inverse) {
    init();
    kernel_inverse = std::make_unique<Eigen::MatrixXd>(
        decomposition->eigenVectors *
        decomposition->eigenValues_inv.asDiagonal() *
        decomposition->eigenVectors.transpose());
  }
  return *kernel_inverse;
};

const KernelMatrix::Decomposition &KernelMatrix::getDecomposition() const {
  init();
  return *this->decomposition;
}

void KernelMatrix::init() const {
  if (nullptr == decomposition) {
    Eigen::EigenSolver<Eigen::MatrixXd> solver(kernel);
    Eigen::MatrixXd eigvectors = solver.eigenvectors().real();
    for (Eigen::Index c = 0; c < eigvectors.cols(); ++c) {
      eigvectors.col(c) /= eigvectors.col(c).norm();
    }
    Eigen::VectorXd eigs = solver.eigenvalues().real();
    Eigen::VectorXd eigs_inv = eigs;
    for (auto &val : eigs_inv) {
      val = 1.0 / val;
    }
    decomposition = std::make_unique<Decomposition>(
        Decomposition{eigvectors, eigs, eigs_inv});
  }
}

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
                       const MatrixIndices &indices) {
  Eigen::MatrixXd result(indices.end - indices.start,
                         indices.end - indices.start);
  Eigen::Index c;
  Eigen::Index pos_row = 0, pos_col;
  for (Eigen::Index r = indices.start; r < indices.end; ++r, ++pos_row) {
    pos_col = pos_row;
    for (c = r; c < indices.end; ++c, ++pos_col) {
      result(pos_row, pos_col) =
          function.evaluate(samples[static_cast<std::size_t>(r)],
                            samples[static_cast<std::size_t>(c)]);
      if (pos_row != pos_col) {
        result(pos_col, pos_row) = result(pos_row, pos_col);
      }
    }
  }
  return result;
}

Eigen::MatrixXd
compute_kernel_portion(const std::vector<Eigen::VectorXd> &samples,
                       const KernelFunction &function,
                       const MatrixIndices &rows, const MatrixIndices &cols) {
  if (rows == cols) {
    return compute_kernel_portion(samples, function, rows);
  }
  Eigen::MatrixXd result(rows.end - rows.start, cols.end - cols.start);
  Eigen::Index c;
  Eigen::Index pos_row = 0, pos_col;
  for (Eigen::Index r = rows.start; r < rows.end; ++r, ++pos_row) {
    pos_col = 0;
    for (c = cols.start; c < cols.end; ++c, ++pos_col) {
      result(pos_row, pos_col) =
          function.evaluate(samples[static_cast<std::size_t>(r)],
                            samples[static_cast<std::size_t>(c)]);
    }
  }
  return result;
}
} // namespace

void KernelAware::updateKernel() {
  const auto &input_samples = getTrainSet()->GetSamplesInput().GetSamples();
  if (nullptr == kernel) {
    kernel.reset(new Kernel{compute_kernel_portion(
        input_samples, *kernelFunction,
        MatrixIndices{0, static_cast<Eigen::Index>(input_samples.size())},
        MatrixIndices{0, static_cast<Eigen::Index>(input_samples.size())})});
  } else {
    MatrixIndices old_indices = MatrixIndices{0, kernel->getKernel().rows()};
    MatrixIndices new_indices =
        MatrixIndices{kernel->getKernel().rows(),
                      static_cast<Eigen::Index>(input_samples.size())};

    Eigen::MatrixXd K_old_new = compute_kernel_portion(
        input_samples, *kernelFunction, old_indices, new_indices);

    Eigen::MatrixXd K_new_new = compute_kernel_portion(
        input_samples, *kernelFunction, new_indices, new_indices);

    Eigen::MatrixXd new_kernel(input_samples.size(), input_samples.size());

    set_matrix_portion(new_kernel, kernel->getKernel(), old_indices,
                       old_indices);
    set_matrix_portion(new_kernel, K_old_new, old_indices, new_indices);
    set_matrix_portion(new_kernel, K_old_new.transpose(), new_indices,
                       old_indices);
    set_matrix_portion(new_kernel, K_new_new, new_indices, new_indices);
    kernel.reset(new Kernel{new_kernel});
  }
}

void KernelAware::resetKernel() { kernel.reset(); }

Eigen::MatrixXd KernelAware::getCovariance() const {
  if (nullptr == kernel) {
    throw gauss::gp::Error("Trying to access null kernel covariance");
  }
  return kernel->getKernel();
};

Eigen::MatrixXd KernelAware::getCovarianceInv() const {
  if (nullptr == kernel) {
    throw gauss::gp::Error("Trying to access null inverse kernel covariance");
  }
  return kernel->getKernelInv();
};

double KernelAware::getCovarianceDeterminant() const {
  if (nullptr == kernel) {
    throw gauss::gp::Error("Trying to access null kernel determinant");
  }
  double result = 1.0;
  for (const auto &eig_val : kernel->getDecomposition().eigenValues) {
    result *= eig_val;
  }
  return result;
};
} // namespace gauss::gp
