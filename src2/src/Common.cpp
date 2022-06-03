/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <Common.h>
#include <GaussianProcess/Error.h>

namespace gauss::gp {
SymmetricMatrixExpandable::SymmetricMatrixExpandable(const Emplacer &emplacer)
    : emplacer(emplacer), matrix(0, 0) {}

namespace {
struct IndexInterval {
  Eigen::Index start;
  Eigen::Index end;
};

void compute_symmetric_block(Eigen::MatrixXd &recipient,
                             const Emplacer &emplacer,
                             const IndexInterval &indices) {
  for (Eigen::Index r = indices.start; r < indices.end; ++r) {
    recipient(r, r) = emplacer(r, r);
    for (Eigen::Index c = r + 1; c < indices.end; ++c) {
      recipient(r, c) = emplacer(r, c);
      recipient(c, r) = recipient(r, c);
    }
  }
}

void compute_asymmetric_block(Eigen::MatrixXd &recipient,
                              const Emplacer &emplacer,
                              const IndexInterval &rows,
                              const IndexInterval &cols) {
  for (Eigen::Index r = rows.start; r < rows.end; ++r) {
    for (Eigen::Index c = cols.start; c < cols.end; ++c) {
      recipient(r, c) = emplacer(r, c);
    }
  }
}
} // namespace

void SymmetricMatrixExpandable::expand(const Eigen::Index new_size) {
  const auto old_size = matrix.rows();
  if (new_size <= old_size) {
    throw Error{"invalid new size for SymmetricMatrixExpandable"};
  }
  Eigen::MatrixXd new_matrix(new_size, new_size);
  if (0 == old_size) {
    compute_symmetric_block(new_matrix, emplacer, IndexInterval{0, new_size});
  } else {
    new_matrix.block(0, 0, old_size, old_size) = matrix;
    compute_symmetric_block(new_matrix, emplacer,
                            IndexInterval{old_size, new_size});
    compute_asymmetric_block(new_matrix, emplacer, IndexInterval{0, old_size},
                             IndexInterval{old_size, new_size});
    new_matrix.block(old_size, 0, new_size - old_size, old_size) =
        new_matrix.block(0, old_size, old_size, new_size - old_size)
            .transpose();
  }
  matrix = std::move(new_matrix);
  throw std::runtime_error{"chiarire se esiste move per MatrixXd"};
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
