/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <Common.h>
#include <GaussianProcess/Error.h>

namespace gauss::gp {
void ResizableMatrix::resize(const Eigen::Index new_size) {
  if (new_size < size) {
    throw Error{"Invalid new size for SymmetricMatrixExpandable"};
  }
  size = new_size;
}

const Eigen::MatrixXd &ResizableMatrix::access() const {
  if (size != computed_portion_size) {
    computed_portion = makeResized();
    computed_portion_size = size;
  }
  return computed_portion;
}

namespace {
struct IndexInterval {
  Eigen::Index start;
  Eigen::Index end;
};

void compute_symmetric_block(Eigen::MatrixXd &recipient,
                             const SymmetricResizableMatrix::Emplacer &emplacer,
                             const IndexInterval &indices) {
  for (Eigen::Index r = indices.start; r < indices.end; ++r) {
    recipient(r, r) = emplacer(r, r);
    for (Eigen::Index c = r + 1; c < indices.end; ++c) {
      recipient(r, c) = emplacer(r, c);
      recipient(c, r) = recipient(r, c);
    }
  }
}

void compute_asymmetric_block(
    Eigen::MatrixXd &recipient,
    const SymmetricResizableMatrix::Emplacer &emplacer,
    const IndexInterval &rows, const IndexInterval &cols) {
  for (Eigen::Index r = rows.start; r < rows.end; ++r) {
    for (Eigen::Index c = cols.start; c < cols.end; ++c) {
      recipient(r, c) = emplacer(r, c);
    }
  }
}
} // namespace

SymmetricResizableMatrix::SymmetricResizableMatrix(const Emplacer &emplacer)
    : emplacer(emplacer) {}

Eigen::MatrixXd SymmetricResizableMatrix::makeResized() const {
  const auto size = getSize();
  const auto computed_size = getComputedSize();
  Eigen::MatrixXd new_matrix = Eigen::MatrixXd::Zero(size, size);
  if (0 == computed_size) {
    compute_symmetric_block(new_matrix, emplacer, IndexInterval{0, size});
  } else {
    new_matrix.block(0, 0, computed_size, computed_size) = getComputedPortion();
    compute_symmetric_block(new_matrix, emplacer,
                            IndexInterval{computed_size, size});
    compute_asymmetric_block(new_matrix, emplacer,
                             IndexInterval{0, computed_size},
                             IndexInterval{computed_size, size});
    new_matrix.block(computed_size, 0, size - computed_size, computed_size) =
        new_matrix.block(0, computed_size, computed_size, size - computed_size)
            .transpose();
  }
  return new_matrix;
}

double trace_product(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
  Eigen::Index size = a.rows();
  double result = 0;
  for (Eigen::Index i = 0; i < size; ++i) {
    result += a.row(i).dot(b.col(i));
  }
  return result;
}

} // namespace gauss::gp
