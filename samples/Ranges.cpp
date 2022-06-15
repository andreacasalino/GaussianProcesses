/**
 * Author:    Andrea Casalino
 * Created:   29.11.2021
 *
 * report any bug to andrecasa91@gmail.com.
 **/

#include "Ranges.h"

namespace gauss::gp::samples {
Linspace::Linspace(const double min, const double max, const std::size_t size)
    : min(min), delta((max - min) / static_cast<double>(size - 1)), i(0),
      size(size) {}

Linspace::Linspace(const Linspace &o)
    : Linspace(min, min + (size - 1) * delta, size) {}

Linspace &Linspace::operator++() {
  ++i;
  return *this;
}

std::vector<Eigen::VectorXd> linspace(const double min, const double max,
                                      const std::size_t size) {
  std::vector<Eigen::VectorXd> result;
  result.reserve(size);
  for (Linspace range(min, max, size); range(); ++range) {
    result.emplace_back(1) << range.eval();
  }
  return result;
}

Grid::Grid(const std::array<double, 2> &min_corner,
           const std::array<double, 2> &max_corner, const std::size_t size)
    : row(min_corner[0], max_corner[0], size) {
  col = std::make_unique<Linspace>(min_corner[1], max_corner[1], size);
}

Eigen::Vector2d Grid::eval() const {
  Eigen::Vector2d result;
  result << row.eval(), col->eval();
  return result;
}

Grid &Grid::operator++() {
  ++(*col);
  if (!(*col)()) {
    col = std::make_unique<Linspace>(*col);
    ++row;
  }
  return *this;
}

std::vector<std::vector<Eigen::Vector2d>>
grid(const std::array<double, 2> &min_corner,
     const std::array<double, 2> &max_corner, const std::size_t size) {
  std::vector<std::vector<Eigen::Vector2d>> result;
  result.resize(size);
  for (std::size_t k = 0; k < size; ++k) {
    result[k].resize(size);
  }
  for (Grid range(min_corner, max_corner, size); range(); ++range) {
    auto indices = range.indices();
    result[indices[0]][indices[1]] = range.eval();
  }
  return result;
}
} // namespace gauss::gp::samples
