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
    : emplacer(emplacer), size(0), computed_portion(0, 0) {}

void SymmetricMatrixExpandable::resize(const Eigen::Index new_size) {
  if (new_size < size) {
    throw Error{"invalid new size for SymmetricMatrixExpandable"};
  }
  size = new_size;
}

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

const Eigen::MatrixXd &SymmetricMatrixExpandable::access() const {
  const auto &computed_portion_size = computed_portion.rows();
  if (computed_portion_size < size) {
    Eigen::MatrixXd new_matrix = Eigen::MatrixXd::Zero(size, size);
    if (0 == computed_portion_size) {
      compute_symmetric_block(new_matrix, emplacer, IndexInterval{0, size});
    } else {
      new_matrix.block(0, 0, computed_portion_size, computed_portion_size) =
          computed_portion;
      compute_symmetric_block(new_matrix, emplacer,
                              IndexInterval{computed_portion_size, size});
      compute_asymmetric_block(new_matrix, emplacer,
                               IndexInterval{0, computed_portion_size},
                               IndexInterval{computed_portion_size, size});
      new_matrix.block(computed_portion_size, 0, size - computed_portion_size,
                       computed_portion_size) =
          new_matrix
              .block(0, computed_portion_size, computed_portion_size,
                     size - computed_portion_size)
              .transpose();
    }
    computed_portion = std::move(new_matrix);
    throw std::runtime_error{"chiarire se esiste move per MatrixXd"};
  }
  return computed_portion;
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
