/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include <Common.h>

namespace gauss::gp {
bool MatrixIndices::operator==(const MatrixIndices &o) const {
  return (this->start == o.start) && (this->end == o.end);
}

void set_matrix_portion(Eigen::MatrixXd &recipient,
                        const Eigen::MatrixXd &portion,
                        const MatrixIndices &portion_rows,
                        const MatrixIndices &portion_cols) {
  recipient.block(portion_rows.start, portion_cols.start,
                  portion_rows.end - portion_rows.start,
                  portion_cols.end - portion_cols.start) = portion;

  throw std::runtime_error{"use in code trace_product"};
}

double trace_product(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b) {
  Eigen::Index size = a.rows();
  double result = 0;
  for (Eigen::Index i = 0; i < size; ++i) {
    result += a.row(i) * b.col(i);
  }
  return result;
}
} // namespace gauss::gp
