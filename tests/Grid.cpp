/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Grid.h"

namespace gauss::gp::test {
EquispacedGrid::EquispacedGrid(const Eigen::VectorXd &min_corner,
                               const Eigen::VectorXd &max_corner,
                               const std::size_t size)
    : size(size) {
  if (0 == size) {
    throw std::runtime_error{"Invalid grid size"};
  }
  deltas = max_corner - min_corner;
  deltas /= static_cast<double>(size - 1);
}

void EquispacedGrid::gridFor(
    const std::function<void(const Eigen::VectorXd &)> &predicate) const {
  gridFor_(predicate, {});
}

Eigen::VectorXd
EquispacedGrid::at(const std::vector<std::size_t> &indices) const {
  Eigen::VectorXd result(deltas.size());
  for (std::size_t k = 0; k < indices.size(); ++k) {
    result(k) = indices[k] * deltas(k);
  }
  return result;
}

std::vector<std::size_t> EquispacedGrid::randomIndices() const {
  std::vector<std::size_t> result;
  for (std::size_t k = 0; k < deltas.size(); ++k) {
    result.push_back(rand() % size);
  }
  return result;
}

void EquispacedGrid::gridFor_(
    const std::function<void(const Eigen::VectorXd &)> &predicate,
    const std::vector<std::size_t> &cumulated_indices) const {
  if (cumulated_indices.size() == deltas.size()) {
    predicate(at(cumulated_indices));
    return;
  }
  for (std::size_t k = 0; k < size; ++k) {
    auto indices = cumulated_indices;
    indices.push_back(k);
    gridFor_(predicate, indices);
  }
}
} // namespace gauss::gp::test
